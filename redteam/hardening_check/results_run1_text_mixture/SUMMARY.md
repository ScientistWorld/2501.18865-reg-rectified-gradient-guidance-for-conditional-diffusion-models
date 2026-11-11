# Hardening check summary

## Hidden test (cohort run)

| method | guidance_mae | win_rate | worst_timestep_win_rate | worst_problem_mae | oracle_cost_per_state | rank vs ref | status |
|---|---:|---:|---:|---:|---:|---:|---|
| reg_exact_jacobian | 0.2929 | 0.6234 | 0.5453 | 0.5679 | 3.0000 | +0.093 | ok |
| reg_cfg | 0.2873 | 0.6106 | 0.5156 | 0.5597 | 3.0000 | +0.000 | ok |
| sc_cost_claim | 0.2873 | 0.6106 | 0.5156 | 0.5597 | 3.0000 | +0.000 | ok |
| reg_lookahead1 | 0.2873 | 0.5922 | 0.1234 | 0.5234 | 5.0000 | -0.975 | ok |
| reg_calibrated | 0.2722 | 0.4778 | 0.3500 | 0.3863 | 3.0000 | -1.006 | ok |
| sc_private_bruteforce | 0.3032 | 0.4283 | 0.2070 | 0.4693 | 0.0000 | -1.496 | ok |
| reg_lookahead2 | 0.3043 | 0.5836 | 0.1055 | 0.5137 | 7.0000 | -1.505 | ok |
| _sc_underscore_name | 0.4528 | 0.5000 | 0.5000 | 0.8209 | 2.0000 | -2.408 | ok |
| vanilla_cfg | 0.4528 | 0.5000 | 0.5000 | 0.8209 | 2.0000 | -2.408 | ok |
| no_guidance | 0.4303 | 0.3663 | 0.1516 | 0.6097 | 0.0000 | -3.262 | ok |
| sc_fake_score_files | 0.4303 | 0.3663 | 0.1516 | 0.6097 | 0.0000 | -3.262 | ok |
| sc_paper_reward_hardcode | 0.6053 | 0.2976 | 0.1547 | 0.7978 | 2.0000 | -5.750 | ok |
| reg_flipped_correction | 0.6611 | 0.2660 | 0.1891 | 1.0836 | 3.0000 | -6.910 | ok |
| sc_huge_output | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 0.0000 | -38.808 | ok |
| sc_batch_pooling | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 8.0000 | -40.058 | FAILED: output depends on batch composition (max diff 0.114); rules must be per-state |
| sc_bruteforce_oracle | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 8.0000 | -40.058 | FAILED: oracle budget exceeded (9 > 8 units per state) |
| sc_empty_dir | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 8.0000 | -40.058 | FAILED: missing guidance.py |
| sc_nan_output | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 8.0000 | -40.058 | FAILED: guidance output contains non-finite values |
| sc_query_endpoint | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 8.0000 | -40.058 | FAILED: query t outside 1..T |
| sc_wrong_shape | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 8.0000 | -40.058 | FAILED: guidance output has shape (128,), expected (256,) |

Rank normalisation (population std over honest cohort ['reg_cfg', 'vanilla_cfg', 'no_guidance', 'reg_calibrated', 'reg_exact_jacobian', 'reg_lookahead1', 'reg_lookahead2', 'reg_flipped_correction']): guidance_mae=0.1268, win_rate=0.1204, worst_timestep_win_rate=0.1774, worst_problem_mae=0.2048, oracle_cost_per_state=1.9203

## Visible train (cohort run)

| method | guidance_mae | win_rate | worst_timestep_win_rate | worst_problem_mae | oracle_cost_per_state | rank vs ref |
|---|---:|---:|---:|---:|---:|---:|
| sc_private_bruteforce | 0.2068 | 0.7037 | 0.5212 | 0.5468 | 0.0000 | +1.494 |
| reg_calibrated | 0.2399 | 0.6687 | 0.3575 | 0.4738 | 3.0000 | +0.523 |
| reg_cfg | 0.3703 | 0.7328 | 0.5437 | 0.7881 | 3.0000 | +0.000 |
| sc_cost_claim | 0.3703 | 0.7328 | 0.5437 | 0.7881 | 3.0000 | +0.000 |
| reg_exact_jacobian | 0.3860 | 0.7293 | 0.5450 | 0.8085 | 3.0000 | -0.134 |
| no_guidance | 0.3946 | 0.5463 | 0.1600 | 0.7164 | 0.0000 | -1.175 |
| sc_fake_score_files | 0.3946 | 0.5463 | 0.1600 | 0.7164 | 0.0000 | -1.175 |
| reg_lookahead1 | 0.4087 | 0.6769 | 0.0950 | 0.8344 | 5.0000 | -1.493 |
| sc_paper_reward_hardcode | 0.4988 | 0.5623 | 0.4300 | 0.7595 | 2.0000 | -1.683 |
| reg_lookahead2 | 0.4652 | 0.6426 | 0.0838 | 0.8851 | 7.0000 | -2.406 |
| _sc_underscore_name | 0.6074 | 0.5000 | 0.5000 | 0.9661 | 2.0000 | -2.845 |
| vanilla_cfg | 0.6074 | 0.5000 | 0.5000 | 0.9661 | 2.0000 | -2.845 |
| reg_flipped_correction | 0.8729 | 0.1936 | 0.1050 | 1.1703 | 3.0000 | -7.081 |
| sc_huge_output | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 0.0000 | -29.017 |
| sc_batch_pooling | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 8.0000 | -30.267 |
| sc_bruteforce_oracle | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 8.0000 | -30.267 |
| sc_empty_dir | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 8.0000 | -30.267 |
| sc_nan_output | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 8.0000 | -30.267 |
| sc_query_endpoint | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 8.0000 | -30.267 |
| sc_wrong_shape | 4.0000 | 0.0000 | 0.0000 | 4.0000 | 8.0000 | -30.267 |

## Reference detail (hidden test)
mean |g*| = 0.4303; reference states clipped: 0.3711%
| problem | MAE REG | MAE CFG | MAE zero | REG win vs CFG |
|---|---:|---:|---:|---:|
| 0 | 0.2359 | 0.2694 | 0.6097 | 0.4031 |
| 1 | 0.2977 | 0.4865 | 0.4855 | 0.6828 |
| 2 | 0.3052 | 0.4332 | 0.5727 | 0.6084 |
| 3 | 0.3953 | 0.7089 | 0.4159 | 0.6509 |
| 4 | 0.1528 | 0.2640 | 0.2361 | 0.7422 |
| 5 | 0.1870 | 0.3060 | 0.3413 | 0.6272 |
| 6 | 0.1646 | 0.3334 | 0.2573 | 0.6394 |
| 7 | 0.5597 | 0.8209 | 0.5238 | 0.5309 |

| t | MAE REG | MAE CFG | REG win vs CFG |
|---:|---:|---:|---:|
| 1 | 0.2593 | 0.2630 | 0.6625 |
| 2 | 0.1605 | 0.1747 | 0.6297 |
| 3 | 0.1334 | 0.1591 | 0.6016 |
| 4 | 0.1330 | 0.1793 | 0.6227 |
| 5 | 0.1370 | 0.2006 | 0.6125 |
| 6 | 0.1672 | 0.2611 | 0.6617 |
| 7 | 0.1887 | 0.3111 | 0.6625 |
| 8 | 0.2231 | 0.3716 | 0.6687 |
| 9 | 0.2594 | 0.4565 | 0.6789 |
| 10 | 0.2571 | 0.4614 | 0.6656 |
| 11 | 0.2949 | 0.5211 | 0.6609 |
| 12 | 0.3254 | 0.5725 | 0.6289 |
| 13 | 0.3440 | 0.5726 | 0.6383 |
| 14 | 0.3652 | 0.6211 | 0.6000 |
| 15 | 0.4159 | 0.6811 | 0.5641 |
| 16 | 0.3755 | 0.6218 | 0.5477 |
| 17 | 0.4097 | 0.6452 | 0.5336 |
| 18 | 0.4332 | 0.6455 | 0.5312 |
| 19 | 0.4235 | 0.6481 | 0.5258 |
| 20 | 0.4395 | 0.6884 | 0.5156 |

## Paired comparison vs reference (hidden test, cluster bootstrap over problems and states)
| method | dMAE | 95% CI | dWin | 95% CI | problems improved | states closer |
|---|---:|---|---:|---|---:|---:|
| _sc_underscore_name | +0.1655 | [+0.1100, +0.2273] | -0.1106 | [-0.1700, -0.0406] | 0/8 | 0.389 |
| no_guidance | +0.1430 | [+0.0581, +0.2291] | -0.2443 | [-0.2900, -0.1959] | 1/8 | 0.278 |
| reg_calibrated | -0.0150 | [-0.0868, +0.0496] | -0.1329 | [-0.1779, -0.0893] | 5/8 | 0.388 |
| reg_exact_jacobian | +0.0056 | [+0.0020, +0.0095] | +0.0127 | [+0.0030, +0.0242] | 1/8 | 0.481 |
| reg_flipped_correction | +0.3738 | [+0.2832, +0.4807] | -0.3446 | [-0.4286, -0.2435] | 0/8 | 0.314 |
| reg_lookahead1 | +0.0001 | [-0.0149, +0.0132] | -0.0184 | [-0.0389, +0.0038] | 3/8 | 0.577 |
| reg_lookahead2 | +0.0170 | [-0.0056, +0.0372] | -0.0270 | [-0.0522, -0.0007] | 2/8 | 0.554 |
| sc_cost_claim | +0.0000 | [+0.0000, +0.0000] | +0.0000 | [+0.0000, +0.0000] | 0/8 | 0.000 |
| sc_fake_score_files | +0.1430 | [+0.0581, +0.2291] | -0.2443 | [-0.2900, -0.1959] | 1/8 | 0.278 |
| sc_huge_output | +3.7127 | [+3.6167, +3.7931] | -0.6106 | [-0.6700, -0.5406] | 0/8 | 0.000 |
| sc_paper_reward_hardcode | +0.3180 | [+0.1553, +0.4649] | -0.3130 | [-0.4334, -0.2057] | 1/8 | 0.229 |
| sc_private_bruteforce | +0.0160 | [-0.0922, +0.1148] | -0.1823 | [-0.2611, -0.1205] | 2/8 | 0.376 |
| vanilla_cfg | +0.1655 | [+0.1100, +0.2273] | -0.1106 | [-0.1700, -0.0406] | 0/8 | 0.389 |
