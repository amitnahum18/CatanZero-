"""Distill native F preferences on recorded TRAINING states, not test games."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import random
from time import perf_counter
from catanatron.players.value import get_value_fn
from catanatron.models.enums import ActionType
from catanatron.serialization import action_to_json,map_to_json
from catanzero.dataset import observation,ordered_actions
from catanzero.features import FeatureEncoder,BUILD_VERSION
from catanzero.search_labels import live_belief
from catanzero.simulation import create_game,SimulationConfig
from catanzero.static_policies import forbid_forward_simulation
from catanzero.tournament import atomic_json


def detached_label_copy(game):
    """Fork a label branch without sharing its random stream with siblings.

    Reusing the same initial RNG state supplies common random draws across
    alternatives. Game.copy alone shares the engine RNG, making stochastic
    teacher labels depend on candidate iteration order.
    """
    child = game.copy()
    branch_rng = random.Random()
    branch_rng.setstate(game.state.random.getstate())
    child.random = child.state.random = branch_rng
    return child


def label_game(source,output,trade_followup=False,build_followup=False):
    output.mkdir(parents=True,exist_ok=False)
    result=json.loads((source/'result.json').read_text())
    game=create_game(SimulationConfig(players=2),result['seed'])
    learner=game.state.colors[result['learner_seat']]
    encoder=FeatureEncoder(map_to_json(game.state.board.map),BUILD_VERSION,allow_fewer_players=True)
    groups=[];start=perf_counter();followups=0;road_followups=0;double_trade_followups=0
    value_fn=get_value_fn('base_fn',None)
    with gzip.open(source/'decisions.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);actions=ordered_actions(game);actor=game.state.current_color()
            assert row['action_index']==len(game.state.action_records)
            inputs={'map_ref':0,'observation':observation(game,actor),'legal_actions':[action_to_json(a) for a in actions]}
            assert {k:row['inputs'][k] for k in inputs}==inputs
            selected=next(a for a in actions if action_to_json(a)==row['inputs']['action'])
            if actor==learner and len(actions)>1:
                before=game.state.random.getstate()
                with forbid_forward_simulation():features=[encoder.encode({**inputs,'action':a}) for a in inputs['legal_actions']]
                seed=result['seed']*1000003+row['action_index']
                view=live_belief(game,actor,seed)
                scores=[]
                for action in actions:
                    child=detached_label_copy(view);child.execute(action)
                    score=value_fn(child,actor)
                    expand=(trade_followup and action.action_type==ActionType.MARITIME_TRADE) or (build_followup and action.action_type==ActionType.BUILD_ROAD)
                    if expand and child.state.current_color()==actor and child.winning_color() is None:
                        # Both operations are deterministic with the observed
                        # hand/public bank. Only the offline labeler simulates.
                        for next_action in ordered_actions(child):
                            if next_action.action_type in (ActionType.BUILD_CITY,ActionType.BUILD_SETTLEMENT):
                                next_child=child.copy();next_child.execute(next_action)
                                score=max(score,value_fn(next_child,actor));followups+=1
                                if action.action_type==ActionType.BUILD_ROAD:road_followups+=1
                            elif build_followup and action.action_type==ActionType.MARITIME_TRADE and next_action.action_type==ActionType.MARITIME_TRADE:
                                second=child.copy();second.execute(next_action)
                                for build in ordered_actions(second):
                                    if build.action_type in (ActionType.BUILD_CITY,ActionType.BUILD_SETTLEMENT):
                                        final=second.copy();final.execute(build)
                                        score=max(score,value_fn(final,actor));double_trade_followups+=1
                    scores.append(score)
                assert game.state.random.getstate()==before and observation(game,actor)==inputs['observation']
                unique=sorted(set(scores));ranks={v:i/max(1,len(unique)-1) for i,v in enumerate(unique)}
                low,high=min(scores),max(scores)
                groups.append({'game_index':result['game_index'],'seed':result['seed'],'action_index':row['action_index'],
                               'features':features,'ordinal':[ranks[s] if len(unique)>1 else .5 for s in scores],
                               'range':[(s-low)/(high-low) if high>low else .5 for s in scores],
                               'teacher_index':max(range(len(scores)),key=scores.__getitem__),'score':scores})
            game.execute(selected)
    assert game.winning_color().value==result['winner']
    with gzip.open(output/'labels.pkl.gz','wb',compresslevel=1) as stream:pickle.dump(groups,stream,protocol=5)
    info={'source':str(source),'game_index':result['game_index'],'seed':result['seed'],
          'initial_map_hash':result['initial_map_hash'],'decision_groups':len(groups),
          'candidate_rows':sum(len(g['features']) for g in groups),'seconds':perf_counter()-start,
          'trade_build_followups':followups,
          'road_build_followups':road_followups,'double_trade_build_followups':double_trade_followups,
          'labeler':'native F on detached public belief; actor known hand only','feature_encoder_guarded':True,
          'label_rng_protocol':'detached_common_stream_per_candidate_v2'}
    atomic_json(output/'result.json',info);return info


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trade-followup',action='store_true')
    parser.add_argument('--build-followup',action='store_true')
    args=parser.parse_args();print(json.dumps(label_game(args.source,args.output,args.trade_followup,args.build_followup)),flush=True)
