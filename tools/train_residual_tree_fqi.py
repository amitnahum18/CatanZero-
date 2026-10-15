"""Fast depth-10 ExtraTrees FQI: fresh two-player games against native F."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shutil
from time import perf_counter

import joblib
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import ExtraTreesRegressor
from train_online_sarsa import REPO, Worker, write
from train_three_game_sarsa import exclusions
from catanzero.duel_protocol import wilson_interval
from catanzero.tree_duel_reward import transition_reward, potential

TREE_PARAMS = {"n_estimators":64, "max_depth":10, "min_samples_leaf":16,
               "max_features":.75, "n_jobs":4}


class ResidualTrees:
    def __init__(self, base, normalizer, forest=None, version=0):
        self.base, self.normalizer, self.forest, self.version = base, normalizer, forest, version
        self.offset, self.scale = np.asarray(normalizer["offset"]), np.asarray(normalizer["scale"])
        self.weights = np.asarray(base["weights"])
        self.names = normalizer["feature_names"]
        self.vp_index = self.names.index("own_actual_vp")
        self.enemy_index = self.names.index("opponent_1_VICTORY_POINTS")
        assert len(self.weights) == len(self.offset) == 264

    def normalize(self, raw):
        return np.clip((np.asarray(raw,dtype=np.float64)-self.offset)/self.scale, -4., 4.)

    def phi(self, raw):
        raw = np.asarray(raw,dtype=np.float64)
        return potential(raw[:,self.vp_index],raw[:,self.enemy_index])

    def anchor(self, raw):
        raw = np.asarray(raw,dtype=np.float64)
        original = self.normalize(raw)@self.weights+self.base["intercept"]
        # Approximate scale conversion from v4's 0/1 outcome to -1/+1, then
        # correct the potential baseline. This state-only shift preserves argmax.
        return 2*original-1+.05*raw[:,self.vp_index]-.02*(raw[:,self.vp_index]-raw[:,self.enemy_index])

    def predict(self, raw):
        result = self.anchor(raw)
        if self.forest is not None and len(result):
            result = result+self.forest.predict(self.normalize(raw))
        return result

    def choose(self, response, rng, epsilon):
        raw = np.asarray(response["features"],dtype=np.float64)
        scores = self.predict(raw)
        explored = rng.random() < epsilon
        selected = rng.randrange(len(raw)) if explored else int(np.argmax(scores))
        return {"index":selected,"raw":raw[selected],"scores":scores.tolist(),
                "version":self.version,"explored":explored}

    def save(self, path):
        joblib.dump(self.forest,path,compress=3)


class RLTreeModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        base=json.loads(Path(context.artifacts["base"]).read_text())
        norm=json.loads(Path(context.artifacts["normalizer"]).read_text())
        forest=joblib.load(context.artifacts["forest"])
        self.scorer=ResidualTrees(base,norm,forest)

    def predict(self, context, model_input, params=None):
        raw=np.asarray(model_input,dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 264 or not np.isfinite(raw).all():
            raise ValueError("expected finite raw predecision features, shape (n,264)")
        return self.scorer.predict(raw)


def play_phase(root, name, model, games, workers, *, training, base_seed, excluded, epsilon=.15,
               record_candidates=False, opponent_policy="F", max_seconds=180):
    directory=root/name
    directory.mkdir()
    is_search_spec = opponent_policy.split(":", 1)[0].upper() in ("AB", "M")
    if opponent_policy not in ("F", "F_static") and not is_search_spec:
        raise ValueError("unsupported opponent policy")
    config={"base_seed":base_seed,"players":2,"opponent_policy":opponent_policy,"feature_version":model.normalizer["feature_version"],
            "learner_policy":getattr(model,"policy_name","Residual_ExtraTrees_FQI"),"max_actions":4000,"max_seconds":max_seconds,
            "trace_compresslevel":1,"excluded_board_hashes":excluded,"training":training,
            "epsilon":epsilon if training else 0.,"model_version":model.version,
            "gamma_per_actual_action":1.,"win_reward":1.,"loss_reward":-1.,
            "vp_potential_weight":.05,"gap_potential_weight":.02}
    if training:config.update(getattr(model,'behavior_config',{}))
    write(directory/"config.json",config)
    started=perf_counter()
    processes=[Worker(directory/"config.json",directory/f"worker_{i}.log") for i in range(min(workers,games))]
    results, samples, slots = [], [], {}
    xs, nexts, ranges, rewards, ds = [], [], [], [], []
    cursor=0
    current_candidates,current_ranges,behavior_probabilities=[],[],[]
    candidate_cursor=0

    def start_game(worker,index):
        response=worker.rpc({"op":"reset","index":index,"output":str(directory/f"game_{index:04d}")})
        assert response["kind"] == "decision"
        rng=random.Random((base_seed+index)*1000003+71)
        return {"worker":worker,"response":response,"rng":rng,
                "plan":model.choose(response,rng,config["epsilon"])}

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool, (directory/"results.jsonl").open("w",encoding="utf-8") as output:
            slots=dict(enumerate(pool.map(lambda pair:start_game(*pair),[(w,i) for i,w in enumerate(processes)])))
            next_index=len(slots)
            while slots:
                keys=sorted(slots)
                calls=[]
                for key in keys:
                    plan=slots[key]["plan"]
                    calls.append((slots[key]["worker"],{"op":"step","selected_index":plan["index"],
                                  "model_version":plan["version"],"scores":plan["scores"],"explored":plan["explored"]}))
                responses=list(pool.map(lambda pair:pair[0].rpc(pair[1]),calls))
                for key,response in zip(keys,responses):
                    slot=slots[key]
                    terminal=response["kind"]=="finished" and response["termination"]=="victory"
                    if training and (terminal or response["kind"]=="decision"):
                        r,d=transition_reward(slot["response"],response,terminal=terminal,won=response.get("learner_won"))
                        following=np.asarray(response["features"],dtype=np.float64) if not terminal else np.empty((0,264))
                        xs.append(slot["plan"]["raw"])
                        nexts.append(following)
                        ranges.append((cursor,cursor+len(following)))
                        cursor+=len(following)
                        rewards.append(r);ds.append(d)
                        samples.append({"phase":name,"game_index":response["game_index"],
                                        "actor":response["learner_color"],"action_index":slot["response"]["action_index"],
                                        "next_action_index":response["action_index"],"selected_index":slot["plan"]["index"],
                                        "terminal":terminal,"won":response.get("learner_won") if terminal else None,
                                        "reward":r,"discount":d,"phi":potential(slot["response"]["vp"],slot["response"]["leader_public_vp"]),
                                        "next_phi":0. if terminal else potential(response["vp"],response["leader_public_vp"]),
                                        "collection_model_version":model.version})
                        if record_candidates:
                            candidates=np.asarray(slot['response']['features'],dtype=np.float64)
                            current_candidates.append(candidates)
                            current_ranges.append((candidate_cursor,candidate_cursor+len(candidates)))
                            candidate_cursor+=len(candidates)
                            probability=slot['plan']['probability']
                            if not 0<probability<=1:raise ValueError('invalid actual behavior probability')
                            behavior_probabilities.append(probability)
                            samples[-1]['behavior_probability']=probability
                    if response["kind"]=="finished":
                        results.append(response);output.write(json.dumps(response)+"\n");output.flush()
                        if len(results)%50==0 or len(results)==games:
                            print(f"{name}: {len(results)}/{games}, wins={sum(r['learner_won'] is True for r in results)}, elapsed={perf_counter()-started:.1f}s",flush=True)
                        if next_index < games:
                            slots[key]=start_game(slot["worker"],next_index);next_index+=1
                        else:
                            del slots[key]
                    else:
                        slot.update(response=response,plan=None)
                ready=[slots[k] for k in sorted(slots) if slots[k]['plan'] is None]
                if ready:
                    if hasattr(model,'choose_many'):
                        plans=model.choose_many([s['response'] for s in ready],[s['rng'] for s in ready],config['epsilon'])
                    else:
                        plans=[model.choose(s['response'],s['rng'],config['epsilon']) for s in ready]
                    for slot,plan in zip(ready,plans):slot['plan']=plan
    finally:
        for worker in processes:worker.close()
    summary={"phase":name,"scheduled":games,"completed":sum(r["termination"]=="victory" for r in results),
             "wins":sum(r["learner_won"] is True for r in results),"elapsed_seconds":perf_counter()-started,
             "model_version":model.version,"training":training,"transitions":len(samples)}
    summary["unfinished"]=games-summary["completed"]
    write(directory/"summary.json",summary)
    data=None
    if training:
        data={"x":np.asarray(xs),"next_x":np.concatenate(nexts),"ranges":np.asarray(ranges,dtype=np.int64),
              "reward":np.asarray(rewards),"discount":np.asarray(ds),
              "phi":np.asarray([s["phi"] for s in samples]),"next_phi":np.asarray([s["next_phi"] for s in samples])}
        if record_candidates:
            data.update(current_x=np.concatenate(current_candidates),current_ranges=np.asarray(current_ranges,dtype=np.int64),
                        behavior_probability=np.asarray(behavior_probabilities))
        np.savez_compressed(directory/"experience.npz",**data)
        (directory/"transitions.jsonl").write_text("".join(json.dumps(s)+"\n" for s in samples),encoding="utf-8")
    return sorted(results,key=lambda r:r["game_index"]),summary,data


def combine(data_sets):
    merged={key:np.concatenate([d[key] for d in data_sets]) for key in data_sets[0] if key != "ranges"}
    cursor=0;ranges=[]
    for data in data_sets:
        ranges.append(data["ranges"]+cursor);cursor+=len(data["next_x"])
    merged["ranges"]=np.concatenate(ranges)
    return merged


def fit_iteration(model, data, path, *, damping=.25):
    started=perf_counter()
    following=model.predict(data["next_x"])
    continuation=np.asarray([following[a:b].max() if b>a else 0. for a,b in data["ranges"]])
    continuation=np.clip(continuation,-1-data["next_phi"],1-data["next_phi"])
    targets=np.clip(data["reward"]+data["discount"]*continuation,-1-data["phi"],1-data["phi"])
    old=model.predict(data["x"])
    mixed=(1-damping)*old+damping*targets
    residual=mixed-model.anchor(data["x"])
    forest=ExtraTreesRegressor(**TREE_PARAMS,random_state=2026+model.version+1)
    forest.fit(model.normalize(data["x"]),residual)
    forest.n_jobs=1
    model.forest=forest;model.version+=1
    path.mkdir()
    model.save(path/"forest.joblib")
    np.savez_compressed(path/"targets.npz",target=targets,previous_q=old,residual_target=residual)
    depths=[tree.tree_.max_depth for tree in forest.estimators_]
    report={"version":model.version,"transitions":len(targets),"next_candidates":len(following),
            "damping":damping,"fit_seconds":perf_counter()-started,"actual_depth_min":min(depths),
            "actual_depth_max":max(depths),"target_min":float(targets.min()),"target_max":float(targets.max()),
            "mean_prediction_change":float(np.mean(np.abs(model.predict(data['x'])-old)))}
    write(path/"fit.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--games",type=int,default=100)
    parser.add_argument("--val-games",type=int,default=50)
    parser.add_argument("--eval-games",type=int,default=100)
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    if any(n<2 or n%2 for n in (args.games,args.val_games,args.eval_games)) or args.rounds<1 or args.workers<1:
        parser.error("even game counts >=2, positive rounds/workers required")
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    source=REPO/"runs/online-sarsa-features-100-v4"
    excluded,manifest=exclusions(source)
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ("linear_model.json","normalizer.json","model.joblib")}
    shutil.copyfile(source/"linear_model.json",root/"base_model.json")
    shutil.copyfile(source/"normalizer.json",root/"normalizer.json")
    model=ResidualTrees(json.loads((root/"base_model.json").read_text()),json.loads((root/"normalizer.json").read_text()))
    config={"algorithm":"damped residual ExtraTrees Fitted Q Iteration with fresh real interaction",
            "players":2,"opponent":"native F on independent public belief","candidate_search":False,
            "input_columns":264,"source":str(source),"source_hashes":hashes,"tree_params":TREE_PARAMS,
            "rounds":args.rounds,"games_per_round":args.games,"iterations_per_round":3,"damping":.25,
            "validation_games":args.val_games,"test_games_per_model":args.eval_games,"workers":args.workers,
            "reward":"terminal win=+1, loss=-1; Phi=.05 own actual VP+.02(own actual VP-opponent PUBLIC VP), terminal Phi=0",
            "gamma_per_actual_action":1.,"unknown_truncated_tails":"excluded, never labelled loss",
            "anchor":"2*v4Q-1+.05 own VP-.02 public gap; same initial action ordering",
            "target_bounds":"[-1-Phi(s),1-Phi(s)] for the undiscounted terminal objective",
            "training_seeds":[46000000+1000*i for i in range(args.rounds)],"validation_seed":45000000,"test_seed":47000000,
            "selection":"best tree validation win count; earliest tie; heldout test never used for selection",
            "normalization_refitted":False,"excluded_board_hashes":excluded}
    write(root/"config.json",config)
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    client=mlflow.MlflowClient();name="CatanZero_Residual_ExtraTrees_FQI"
    exp=client.get_experiment_by_name(name)
    eid=exp.experiment_id if exp else client.create_experiment(name,artifact_location=(REPO/"runs/mlflow/artifacts").as_uri())
    mlflow.set_experiment(experiment_id=eid)
    started=perf_counter();history=[];data_sets=[];training_rows=[];validations=[]
    with mlflow.start_run(run_name="depth10_nativeF_duel_300_games_symmetric_reward") as run:
        write(root/"tracking.json",{"run_id":run.info.run_id,"tracking_uri":manifest["tracking_uri"]})
        mlflow.log_params({"model":"ExtraTrees residual FQI","max_depth":10,"trees":64,"training_games":args.rounds*args.games,
                           "opponent":"F","players":2,"candidate_forward_simulation":False,"gamma":1.,"terminal_loss":-1.,"workers":args.workers})
        mlflow.log_artifact(str(root/"config.json"),"protocol_before_training")
        print("STAGE: frozen v4 validation baseline",flush=True)
        val_rows,base_val,_=play_phase(root,"baseline_validation",model,args.val_games,args.workers,training=False,
                                     base_seed=45000000,excluded=excluded)
        val_boards=[r["initial_map_hash"] for r in val_rows]
        for i in range(args.rounds):
            print(f"STAGE: real training round {i+1}/{args.rounds}",flush=True)
            rows,summary,data=play_phase(root,f"training_round_{i+1}",model,args.games,args.workers,training=True,
                                         base_seed=config["training_seeds"][i],excluded=excluded+val_boards+[r['initial_map_hash'] for r in training_rows],
                                         epsilon=.2-.05*i/max(1,args.rounds-1))
            training_rows.extend(rows);data_sets.append(data);merged=combine(data_sets)
            for _ in range(3):
                info=fit_iteration(model,merged,root/f"update_{model.version+1:03d}")
                history.append(info)
                print("FIT: "+json.dumps(info),flush=True)
            _,validation,_=play_phase(root,f"validation_round_{i+1}",model,args.val_games,args.workers,training=False,
                                      base_seed=45000000,excluded=excluded+[r['initial_map_hash'] for r in training_rows])
            validations.append(validation)
            write(root/"training_history.json",history);write(root/"validation_history.json",validations)
        selected=max(range(len(validations)),key=lambda i:(validations[i]["wins"],-i))
        selected_version=validations[selected]["model_version"]
        model.forest=joblib.load(root/f"update_{selected_version:03d}/forest.joblib");model.version=selected_version
        model.save(root/"forest.joblib")
        write(root/"selection.json",{"selected_round":selected+1,"model_version":selected_version,"baseline_validation":base_val,
                                      "tree_validation":validations,"reason":"highest frozen validation wins; earliest tie"})
        print("STAGE: frozen heldout paired test, v4 then selected trees",flush=True)
        test_excluded=excluded+val_boards+[r['initial_map_hash'] for r in training_rows]
        baseline=ResidualTrees(model.base,model.normalizer)
        before_rows,before,_=play_phase(root,"baseline_test",baseline,args.eval_games,args.workers,training=False,
                                        base_seed=47000000,excluded=test_excluded)
        after_rows,after,_=play_phase(root,"tree_test",model,args.eval_games,args.workers,training=False,
                                     base_seed=47000000,excluded=test_excluded)
        plus=minus=paired=0
        for a,b in zip(before_rows,after_rows):
            for k in ("seed","learner_seat","learner_color","initial_map_hash","initial_stream_hash"):assert a[k]==b[k]
            if a["termination"]==b["termination"]=="victory":
                paired+=1;plus+=b["learner_won"] and not a["learner_won"];minus+=a["learner_won"] and not b["learner_won"]
        assert hashes=={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in hashes}
        sample=data_sets[-1]["x"][::max(1,len(data_sets[-1]['x'])//512)]
        expected=model.predict(sample)
        info=mlflow.pyfunc.log_model(name="model",python_model=RLTreeModel(),
              artifacts={"base":str(root/"base_model.json"),"normalizer":str(root/"normalizer.json"),"forest":str(root/"forest.joblib")},
              input_example=sample[:3],signature=infer_signature(sample,expected),
              pip_requirements=["mlflow=="+mlflow.__version__,"numpy=="+np.__version__,"scikit-learn==1.7.1","joblib=="+joblib.__version__])
        restored=mlflow.pyfunc.load_model(info.model_uri)
        reload_error=float(np.max(np.abs(restored.predict(sample)-expected)));assert reload_error<1e-12
        registered=mlflow.register_model(info.model_uri,"CatanZero_Residual_ExtraTrees_FQI")
        client.set_registered_model_alias("CatanZero_Residual_ExtraTrees_FQI","depth10_pilot",str(registered.version))
        write(root/"registration.json",{"family":"CatanZero_Residual_ExtraTrees_FQI","version":str(registered.version),
              "alias":"depth10_pilot","model_uri":info.model_uri,"reload_max_abs_error":reload_error,"raw_input_columns":264})
        report={"training_games":len(training_rows),"training_wins_exploratory":sum(r['learner_won'] is True for r in training_rows),
                "training_transitions":len(merged['x']),"fit_updates":len(history),"fit_seconds_total":sum(h['fit_seconds'] for h in history),
                "selected_round":selected+1,"selected_version":selected_version,"baseline_test":before,"tree_test":after,
                "paired_complete":paired,"gained_wins":int(plus),"lost_wins":int(minus),
                "paired_p":float(binomtest(plus,plus+minus,.5).pvalue) if plus+minus else 1.,
                "tree_wilson_95":wilson_interval(after['wins'],args.eval_games) if not after['unfinished'] else None,
                "reached_50_percent_observed":not after['unfinished'] and after['wins']>=args.eval_games/2,
                "source_unchanged":True,"normalization_unchanged":True,"reload_max_abs_error":reload_error,
                "elapsed_seconds":perf_counter()-started,"forward_simulation_candidate":False,
                "limitations":["short pilot, not a guarantee of 50% or universal strength","two-player base engine, not official variant rules",
                               "max backup over sampled-training coverage can overestimate unsupported actions","v4/norm transferred from four-player training",
                               "depth10 is feature-tree depth, not forward game search","Q predictions are not calibrated win probabilities"]}
        write(root/"report.json",report)
        for file in ("tools/train_residual_tree_fqi.py","src/catanzero/online_env.py","src/catanzero/tree_duel_reward.py"):
            dest=root/"implementation_snapshot"/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/file,dest)
        mlflow.log_metrics({"baseline_test_wins":before['wins'],"tree_test_wins":after['wins'],"paired_p":report['paired_p'],
                            "training_transitions":report['training_transitions'],"fit_seconds_total":report['fit_seconds_total'],
                            "elapsed_seconds":report['elapsed_seconds'],"reload_max_abs_error":reload_error})
        for name in ("config.json","selection.json","training_history.json","validation_history.json","report.json","registration.json"):
            mlflow.log_artifact(str(root/name),"experiment")
        mlflow.log_artifacts(str(root/"implementation_snapshot"),"implementation_snapshot")
        print("RESULT: "+json.dumps(report),flush=True)


if __name__=="__main__":main()
