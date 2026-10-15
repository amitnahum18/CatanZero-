# Archived depth10 residual FQI pilot

300 real two-player training games against native F; 36,379 causal transitions, 9 batch fits, 64 ExtraTrees per fit, maximum depth10. Terminal +1/-1, gamma1, exact potential shaping as documented in the final experiment. Frozen final test: 11/100 versus v4 23/100 on paired boards (p=.01690). No claim of improvement or promotion over v4.

This pilot led to teacher ranking/real-return MC/tree-policy optimization, documented in `tree-policy-depth10-native-f.md`. Local native and MLflow models remain saved. Initial fitting/scoring source tests passed; independent complete replay/MC numerical audits apply to the later final policy and are not retroactively claimed for this archived FQI pilot.
