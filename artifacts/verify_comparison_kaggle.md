# Benchmark cross-check: Kaggle against the local machine

The local machine has confirmed faulty RAM, so every number it produced is suspect. The engine's analytical suite is deterministic in fp64, so running it here and comparing metric by metric says whether bit flips reached the results.

**What counts as evidence.** Only the double-precision physics metrics (188 of 198). Wall-clock times (2) measure the machine, not the arithmetic. Single-precision metrics (8) carry ~1e-7 round-off per operation over ~40,000 steps and are judged at 0.01 rather than 0.001. Where a metric's own sensitivity to a one-bit change in the initial condition has been measured, a difference below that is the scheme amplifying round-off, which says nothing about memory.

**Verdict: this machine reproduces the local benchmark numbers to the precision they are reported at.**

- metrics compared: 198 (188 fp64, 8 fp32, 2 hardware)
- identical: 44
- both below the reporting floor (1e-12), so uninformative: 27
- round-off only (<= 1e-06 relative): 121
- differing beyond round-off but within their class tolerance: 2
- within a measured one-ulp sensitivity: 2
- **meaningful differences or missing: 0**
- pass/fail flips: none

Local: cuda / fp64, commit ea3006a-dirty. Here: Tesla T4 / torch 2.10.0+cu128, commit e9ad14d.

| metric | class | local | Kaggle | relative difference | verdict |
|---|---|---|---|---|---|
| 8.fp32_wall_s | hardware | 294.0445108999993 | 492.96445884099967 | 4.04e-01 | hardware (not evidence) |
| 8.fp64_wall_s | hardware | 403.15234220000275 | 487.3935435960002 | 1.73e-01 | hardware (not evidence) |
| 4.field_o2_relL2_per_period[2] | fp64 | 0.01610363914482908 | 0.016242998482245193 | 8.58e-03 | within its own measured round-off sensitivity |
| 4.field_o2_relL2_per_period[1] | fp64 | 0.012532528823669675 | 0.012482468647533164 | 3.99e-03 | within its own measured round-off sensitivity |
| 8.fp32_mass_err | fp32 | 7.004548298571471e-05 | 7.013599531868293e-05 | 1.29e-03 | differs, within the fp32 tolerance |
| 4.field_o2_relL2_per_period[0] | fp64 | 0.009851190312498745 | 0.00985106020240338 | 1.32e-05 | differs |
| 8.fp32_exchanged_2d_to_1d_m3 | fp32 | 1186244.3790665015 | 1186244.8073330077 | 3.61e-07 | round-off |
| 8.fp32_stored_m3 | fp32 | 2264208.2042103293 | 2264207.936037879 | 1.18e-07 | round-off |
| 8.fp32_exchanged_1d_to_2d_m3 | fp32 | 428480.70711263147 | 428480.66548549855 | 9.72e-08 | round-off |
| 8.fp32_infiltration_m3 | fp32 | 2647772.8933102353 | 2647772.73657084 | 5.92e-08 | round-off |
| 8.fp32_rain_m3 | fp32 | 8153853.28807272 | 8153853.171434577 | 1.43e-08 | round-off |
| 2b.o2_L1[3] | fp64 | 0.001171922956710355 | 0.0011719229567290202 | 1.59e-11 | round-off |
| 2b.o2_order_L1[2] | fp64 | 1.9856513390050656 | 1.985651338985549 | 9.83e-12 | round-off |
| 2b.o2_L1[2] | fp64 | 0.004641300372205641 | 0.004641300372216777 | 2.40e-12 | round-off |
| 2b.o2_order_L1[1] | fp64 | 1.8841801854150475 | 1.8841801854120679 | 1.58e-12 | round-off |
| 2.o2_order_L2[2] | fp64 | -0.011516120023197749 | -0.011516120023210825 | 1.14e-12 | round-off |
| 6.o1_equilibrium_rel_error | fp64 | 0.0001390365622324155 | 0.00013903656223257163 | 1.12e-12 | round-off |
| 6.o2_equilibrium_rel_error | fp64 | 0.000253156108985797 | 0.00025315610898564087 | 6.17e-13 | round-off |
| 2b.o2_L1[0] | fp64 | 0.05536997134405204 | 0.055369971344085035 | 5.96e-13 | round-off |
| 2b.o2_L1[1] | fp64 | 0.017133039996037644 | 0.017133039996043365 | 3.34e-13 | round-off |
| 2b.o2_order_L1[0] | fp64 | 1.6923226169063967 | 1.6923226169067747 | 2.23e-13 | round-off |
| 3.o2_order_L2[2] | fp64 | 0.49723183754960915 | 0.4972318375495229 | 1.73e-13 | round-off |
| 2b.o1_L1[3] | fp64 | 0.03438327832161992 | 0.03438327832162403 | 1.19e-13 | round-off |
| 2b.o1_order_L1[1] | fp64 | 0.9052630647673782 | 0.9052630647672755 | 1.13e-13 | round-off |
| 2.o2_order_L2[1] | fp64 | 0.1407041842220582 | 0.14070418422207154 | 9.49e-14 | round-off |
| 2b.o1_order_L1[2] | fp64 | 0.9488463041555996 | 0.9488463041555153 | 8.88e-14 | round-off |
| 3.o2_L2[3] | fp64 | 0.004710226287901475 | 0.004710226287901873 | 8.45e-14 | round-off |
| 2b.o1_order_L1[0] | fp64 | 0.8263945814886058 | 0.8263945814886741 | 8.26e-14 | round-off |
| 3.isolated_o1_order_L1[2] | fp64 | 0.8026663670100812 | 0.8026663670101367 | 6.92e-14 | round-off |
| 3.o1_order_L1[2] | fp64 | 0.8026663670100812 | 0.8026663670101367 | 6.92e-14 | round-off |
| 3.o2_order_L2[1] | fp64 | 0.8081487672055632 | 0.8081487672055133 | 6.17e-14 | round-off |
| 2b.o1_L1[2] | fp64 | 0.06637101840550856 | 0.06637101840551261 | 6.11e-14 | round-off |
| 2.isolated_o2_order_L1[2] | fp64 | 0.9989748088693354 | 0.9989748088693866 | 5.12e-14 | round-off |
| 2.o2_order_L1[1] | fp64 | 0.43714615248744726 | 0.43714615248746663 | 4.43e-14 | round-off |
| 2.o1_order_L1[2] | fp64 | 0.4873499822740967 | 0.4873499822740768 | 4.09e-14 | round-off |
| 3.isolated_o1_L1[3] | fp64 | 0.0025438034741661992 | 0.002543803474166099 | 3.94e-14 | round-off |
| 3.o1_L1[3] | fp64 | 0.0025438034741661992 | 0.002543803474166099 | 3.94e-14 | round-off |
| 2b.o1_L1[0] | fp64 | 0.22042425179714767 | 0.22042425179715586 | 3.71e-14 | round-off |
| 2.isolated_o1_order_L1[2] | fp64 | 0.7522000336007978 | 0.752200033600771 | 3.56e-14 | round-off |
| 3.o1_order_L2[2] | fp64 | 0.5257220675945286 | 0.5257220675945115 | 3.25e-14 | round-off |
| 2.o2_order_L2[0] | fp64 | 0.5002015612016061 | 0.500201561201593 | 2.62e-14 | round-off |
| 3.o2_L2[2] | fp64 | 0.006648496888384471 | 0.006648496888384635 | 2.47e-14 | round-off |
| 3.isolated_o2_order_L1[0] | fp64 | 0.9769777172797367 | 0.9769777172797139 | 2.33e-14 | round-off |
| 3.o2_order_L1[0] | fp64 | 0.9769777172797367 | 0.9769777172797139 | 2.33e-14 | round-off |
| 3.isolated_o2_L1[2] | fp64 | 0.0011845584425100556 | 0.0011845584425100827 | 2.29e-14 | round-off |
| 3.o2_L1[2] | fp64 | 0.0011845584425100556 | 0.0011845584425100827 | 2.29e-14 | round-off |
| 2.isolated_o2_L1[3] | fp64 | 0.0006354440544792139 | 0.0006354440544792003 | 2.13e-14 | round-off |
| 3.isolated_o2_order_L1[1] | fp64 | 1.0708802342588406 | 1.0708802342588208 | 1.85e-14 | round-off |
| 3.o2_order_L1[1] | fp64 | 1.0708802342588406 | 1.0708802342588208 | 1.85e-14 | round-off |
| 2.o1_order_L2[2] | fp64 | 0.3792975902051983 | 0.37929759020519166 | 1.76e-14 | round-off |
| 3.isolated_o2_L1[3] | fp64 | 0.0005924483198265084 | 0.0005924483198265182 | 1.67e-14 | round-off |
| 3.o2_L1[3] | fp64 | 0.0005924483198265084 | 0.0005924483198265182 | 1.67e-14 | round-off |
| 2.o2_order_L1[0] | fp64 | 0.6766955759854877 | 0.6766955759854772 | 1.54e-14 | round-off |
| 2.isolated_o2_L1[2] | fp64 | 0.0012699853260672803 | 0.0012699853260672983 | 1.42e-14 | round-off |
| 2.isolated_o2_order_L1[1] | fp64 | 0.998557338273468 | 0.9985573382734543 | 1.38e-14 | round-off |
| 2.isolated_o1_L1[3] | fp64 | 0.003224284764158051 | 0.0032242847641580944 | 1.35e-14 | round-off |
| 3.o2_order_L2[0] | fp64 | 0.4259784781344063 | 0.42597847813441203 | 1.34e-14 | round-off |
| 3.o1_L2[3] | fp64 | 0.008198910121867846 | 0.008198910121867952 | 1.29e-14 | round-off |
| 2.o1_L1[3] | fp64 | 0.004831513915866772 | 0.004831513915866832 | 1.24e-14 | round-off |
| 6.o2_rising_limb_nrmse | fp64 | 0.00818371428512319 | 0.008183714285123279 | 1.08e-14 | round-off |
| 2.o2_L1[3] | fp64 | 0.0023161073347202905 | 0.002316107334720266 | 1.07e-14 | round-off |
| 2.o2_L1[2] | fp64 | 0.002761953548611918 | 0.002761953548611889 | 1.05e-14 | round-off |
| 2b.o1_L1[1] | fp64 | 0.12430534922850292 | 0.12430534922850164 | 1.03e-14 | round-off |
| 2.isolated_o2_order_L1[0] | fp64 | 1.013278890099043 | 1.0132788900990328 | 1.01e-14 | round-off |
| 3.o2_L2[1] | fp64 | 0.011641273421108423 | 0.011641273421108309 | 9.83e-15 | round-off |
| 3.isolated_o2_L1[1] | fp64 | 0.002488419320577253 | 0.002488419320577276 | 9.24e-15 | round-off |
| 3.o2_L1[1] | fp64 | 0.002488419320577253 | 0.002488419320577276 | 9.24e-15 | round-off |
| 3.isolated_o2_order_L1[2] | fp64 | 0.9995881624024199 | 0.9995881624024288 | 8.89e-15 | round-off |
| 3.o2_order_L1[2] | fp64 | 0.9995881624024199 | 0.9995881624024288 | 8.89e-15 | round-off |
| 2.o1_order_L1[0] | fp64 | 0.565887395182055 | 0.5658873951820598 | 8.44e-15 | round-off |
| 2.o2_L2[1] | fp64 | 0.005726495270716238 | 0.005726495270716285 | 8.18e-15 | round-off |
| 2.o2_L2[3] | fp64 | 0.005235997421201615 | 0.005235997421201657 | 7.95e-15 | round-off |
| 3.o1_order_L2[0] | fp64 | 0.4182245915190431 | 0.41822459151903996 | 7.57e-15 | round-off |
| 2.o1_order_L2[1] | fp64 | 0.4849450366726047 | 0.484945036672601 | 7.55e-15 | round-off |
| 2.o1_L2[3] | fp64 | 0.007219082209641675 | 0.007219082209641725 | 6.97e-15 | round-off |
| 3.isolated_o2_L1[0] | fp64 | 0.0048980494189044144 | 0.004898049418904382 | 6.55e-15 | round-off |
| 3.o2_L1[0] | fp64 | 0.0048980494189044144 | 0.004898049418904382 | 6.55e-15 | round-off |
| 2.isolated_o1_L1[1] | fp64 | 0.008907852970893408 | 0.008907852970893353 | 6.23e-15 | round-off |
| 3.o2_L2[0] | fp64 | 0.01563985740669242 | 0.01563985740669233 | 5.77e-15 | round-off |
| 3.isolated_o1_order_L1[0] | fp64 | 0.7718176224066976 | 0.7718176224066936 | 5.18e-15 | round-off |
| 3.o1_order_L1[0] | fp64 | 0.7718176224066976 | 0.7718176224066936 | 5.18e-15 | round-off |
| 2.isolated_o1_L1[2] | fp64 | 0.005430854454139359 | 0.005430854454139332 | 4.95e-15 | round-off |
| 2.isolated_o2_L1[1] | fp64 | 0.002537432009638281 | 0.002537432009638293 | 4.61e-15 | round-off |
| 2.o2_L1[0] | fp64 | 0.005977452599511156 | 0.00597745259951113 | 4.35e-15 | round-off |
| 2.isolated_o1_L1[0] | fp64 | 0.014135894632199003 | 0.014135894632198942 | 4.30e-15 | round-off |
| 2.isolated_o1_order_L1[0] | fp64 | 0.6662135403670507 | 0.6662135403670535 | 4.17e-15 | round-off |
| 2.o1_order_L2[0] | fp64 | 0.5328781260035513 | 0.5328781260035531 | 3.33e-15 | round-off |
| 2.o1_order_L1[1] | fp64 | 0.532951285888089 | 0.5329512858880875 | 2.92e-15 | round-off |
| 2.o2_L1[1] | fp64 | 0.0037394736263691363 | 0.003739473626369147 | 2.90e-15 | round-off |
| 3.isolated_o1_order_L1[1] | fp64 | 0.8308692052927422 | 0.8308692052927443 | 2.54e-15 | round-off |
| 3.o1_order_L1[1] | fp64 | 0.8308692052927422 | 0.8308692052927443 | 2.54e-15 | round-off |
| 2.isolated_o1_order_L1[1] | fp64 | 0.7138985460028272 | 0.7138985460028254 | 2.49e-15 | round-off |
| 2.o1_L2[2] | fp64 | 0.00938992102170101 | 0.009389921021701032 | 2.40e-15 | round-off |
| 2.isolated_o2_L1[0] | fp64 | 0.005121789837349766 | 0.005121789837349754 | 2.37e-15 | round-off |
| 3.isolated_o1_L1[0] | fp64 | 0.013476186662658186 | 0.013476186662658155 | 2.32e-15 | round-off |
| 3.o1_L1[0] | fp64 | 0.013476186662658186 | 0.013476186662658155 | 2.32e-15 | round-off |
| 2.o1_L1[1] | fp64 | 0.009799964919612944 | 0.009799964919612922 | 2.30e-15 | round-off |
| 3.o1_L2[0] | fp64 | 0.025408615728381285 | 0.025408615728381247 | 1.50e-15 | round-off |
| 6.o1_rising_limb_nrmse | fp64 | 0.12490251025993257 | 0.12490251025993274 | 1.33e-15 | round-off |
| 2.o1_L1[2] | fp64 | 0.006773142265764339 | 0.00677314226576433 | 1.28e-15 | round-off |
| 8.fp64_exchanged_1d_to_2d_m3 | fp64 | 428538.37319170724 | 428538.37319170777 | 1.22e-15 | round-off |
| 2.o2_L2[2] | fp64 | 0.0051943681438671434 | 0.005194368143867137 | 1.17e-15 | round-off |
| 4.field_o1_relL2_per_period[1] | fp64 | 0.2202983831677182 | 0.22029838316771844 | 1.13e-15 | round-off |
| 2.o1_L2[0] | fp64 | 0.019013288659087264 | 0.019013288659087284 | 1.09e-15 | round-off |
| 3.o1_L2[2] | fp64 | 0.011803593303726274 | 0.011803593303726287 | 1.03e-15 | round-off |
| 3.isolated_o1_L1[2] | fp64 | 0.004437212309315786 | 0.004437212309315782 | 9.77e-16 | round-off |
| 3.o1_L1[2] | fp64 | 0.004437212309315786 | 0.004437212309315782 | 9.77e-16 | round-off |
| 3.o1_order_L2[1] | fp64 | 0.6878670691487089 | 0.6878670691487082 | 9.68e-16 | round-off |
| 2.o1_L1[0] | fp64 | 0.014506866057006807 | 0.014506866057006821 | 9.57e-16 | round-off |
| 2.o2_order_L1[2] | fp64 | 0.253986942816256 | 0.2539869428162562 | 8.74e-16 | round-off |
| 2.o2_L2[0] | fp64 | 0.008099618808196207 | 0.0080996188081962 | 8.57e-16 | round-off |
| 4.field_o1_relL2_per_period[0] | fp64 | 0.11893396986943054 | 0.11893396986943064 | 8.17e-16 | round-off |
| 4.field_o1_relL2_per_period[2] | fp64 | 0.3124968607948186 | 0.3124968607948188 | 7.11e-16 | round-off |
| 3.o1_L2[1] | fp64 | 0.01901441017489242 | 0.01901441017489243 | 5.47e-16 | round-off |
| 8.fp64_infiltration_m3 | fp64 | 2647771.394965788 | 2647771.3949657865 | 5.28e-16 | round-off |
| 8.fp64_rain_m3 | fp64 | 8153993.364876904 | 8153993.3648769 | 4.57e-16 | round-off |
| 3.isolated_o1_L1[1] | fp64 | 0.007892721203112256 | 0.00789272120311226 | 4.40e-16 | round-off |
| 3.o1_L1[1] | fp64 | 0.007892721203112256 | 0.00789272120311226 | 4.40e-16 | round-off |
| 8.fp64_exchanged_2d_to_1d_m3 | fp64 | 1186291.5112686679 | 1186291.5112686674 | 3.93e-16 | round-off |
| 7.exchanged_2d_to_1d_m3 | fp64 | 127094.44141398181 | 127094.44141398185 | 3.43e-16 | round-off |
| 7.exchanged_1d_to_2d_m3 | fp64 | 127149.20842896267 | 127149.20842896271 | 3.43e-16 | round-off |
| 8.fp64_stored_m3 | fp64 | 2264214.1408947124 | 2264214.140894713 | 2.06e-16 | round-off |
| 4.lab_o2_hdry1e-6_relL2_per_period[0] | fp64 | 0.009554223040642781 | 0.009554223040642783 | 1.82e-16 | round-off |
| 4.lab_o2_relL2_per_period[0] | fp64 | 0.052247642977089045 | 0.05224764297708905 | 1.33e-16 | round-off |
| 2.o1_L2[1] | fp64 | 0.0131415000708866 | 0.013141500070886599 | 1.32e-16 | round-off |
| 7.final_2d_volume_m3 | fp64 | 54.76701498075154 | 54.76701498075155 | 1.30e-16 | round-off |
| 7.peak_2d_volume_m3 | fp64 | 124943.75767637728 | 124943.7576763773 | 1.16e-16 | round-off |
| 1.o1_mass_err | fp64 | 1.9564788851534466e-28 | 1.0793544001135703e-27 | n/a | both negligible |
| 1.o1_max_abs_u | fp64 | 2.3060683697684247e-14 | 2.354249286294016e-14 | n/a | both negligible |
| 1.o1_max_abs_v | fp64 | 1.7842420590203678e-14 | 2.0003842327108822e-14 | n/a | both negligible |
| 1.o1_max_eta_dev | fp64 | 2.220446049250313e-15 | 1.7763568394002505e-15 | n/a | both negligible |
| 1.o1_wet_fraction | fp64 | 0.42478461538461537 | 0.42478461538461537 | 0.00e+00 | identical |
| 1.o2_mass_err | fp64 | 4.232523382930428e-28 | 1.6602290561746296e-27 | n/a | both negligible |
| 1.o2_max_abs_u | fp64 | 2.5884776282462563e-14 | 3.131166694024935e-14 | n/a | both negligible |
| 1.o2_max_abs_v | fp64 | 2.821669235806383e-14 | 2.889948278420591e-14 | n/a | both negligible |
| 1.o2_max_eta_dev | fp64 | 2.220446049250313e-15 | 1.7763568394002505e-15 | n/a | both negligible |
| 1.o2_wet_fraction | fp64 | 0.42478461538461537 | 0.42478461538461537 | 0.00e+00 | identical |
| 1.passed | fp64 | 1.0 | 1.0 | 0.00e+00 | identical |
| 2.grids[0] | fp64 | 200.0 | 200.0 | 0.00e+00 | identical |
| 2.grids[1] | fp64 | 400.0 | 400.0 | 0.00e+00 | identical |
| 2.grids[2] | fp64 | 800.0 | 800.0 | 0.00e+00 | identical |
| 2.grids[3] | fp64 | 1600.0 | 1600.0 | 0.00e+00 | identical |
| 2.o1_mass_err | fp64 | 0.0 | 0.0 | n/a | both negligible |
| 2.o2_mass_err | fp64 | 0.0 | 4.440892084622838e-16 | n/a | both negligible |
| 2.passed | fp64 | 1.0 | 1.0 | 0.00e+00 | identical |
| 2b.grids[0] | fp64 | 100.0 | 100.0 | 0.00e+00 | identical |
| 2b.grids[1] | fp64 | 200.0 | 200.0 | 0.00e+00 | identical |
| 2b.grids[2] | fp64 | 400.0 | 400.0 | 0.00e+00 | identical |
| 2b.grids[3] | fp64 | 800.0 | 800.0 | 0.00e+00 | identical |
| 2b.passed | fp64 | 1.0 | 1.0 | 0.00e+00 | identical |
| 2b.reference_grid | fp64 | 6400.0 | 6400.0 | 0.00e+00 | identical |
| 3.grids[0] | fp64 | 200.0 | 200.0 | 0.00e+00 | identical |
| 3.grids[1] | fp64 | 400.0 | 400.0 | 0.00e+00 | identical |
| 3.grids[2] | fp64 | 800.0 | 800.0 | 0.00e+00 | identical |
| 3.grids[3] | fp64 | 1600.0 | 1600.0 | 0.00e+00 | identical |
| 3.o1_mass_err | fp64 | 0.0 | 0.0 | n/a | both negligible |
| 3.o2_mass_err | fp64 | 4.4408920832350593e-16 | 8.881784166470119e-16 | n/a | both negligible |
| 3.passed | fp64 | 1.0 | 1.0 | 0.00e+00 | identical |
| 4.field_o1_clip_m3 | fp64 | 0.0 | 0.0 | n/a | both negligible |
| 4.field_o1_dx_m | fp64 | 20.0 | 20.0 | 0.00e+00 | identical |
| 4.field_o1_mass_err | fp64 | 0.0 | 1.4822240282185291e-16 | n/a | both negligible |
| 4.field_o1_n | fp64 | 200.0 | 200.0 | 0.00e+00 | identical |
| 4.field_o1_period_s | fp64 | 1003.0333403553236 | 1003.0333403553236 | 0.00e+00 | identical |
| 4.field_o2_clip_m3 | fp64 | 0.0 | 0.0 | n/a | both negligible |
| 4.field_o2_dx_m | fp64 | 20.0 | 20.0 | 0.00e+00 | identical |
| 4.field_o2_mass_err | fp64 | 1.4822240282185291e-16 | 1.4822240282185291e-16 | n/a | both negligible |
| 4.field_o2_n | fp64 | 200.0 | 200.0 | 0.00e+00 | identical |
| 4.field_o2_period_s | fp64 | 1003.0333403553236 | 1003.0333403553236 | 0.00e+00 | identical |
| 4.lab_o2_clip_m3 | fp64 | 0.0 | 0.0 | n/a | both negligible |
| 4.lab_o2_dx_m | fp64 | 0.02 | 0.02 | 0.00e+00 | identical |
| 4.lab_o2_hdry1e-6_clip_m3 | fp64 | 0.0 | 0.0 | n/a | both negligible |
| 4.lab_o2_hdry1e-6_dx_m | fp64 | 0.02 | 0.02 | 0.00e+00 | identical |
| 4.lab_o2_hdry1e-6_mass_err | fp64 | 0.0 | 0.0 | n/a | both negligible |
| 4.lab_o2_hdry1e-6_n | fp64 | 200.0 | 200.0 | 0.00e+00 | identical |
| 4.lab_o2_hdry1e-6_period_s | fp64 | 4.485701465466374 | 4.485701465466374 | 0.00e+00 | identical |
| 4.lab_o2_hdry1e-6_relL2_per_period[1] | fp64 | 0.012622710520427276 | 0.012622710520427276 | 0.00e+00 | identical |
| 4.lab_o2_hdry1e-6_relL2_per_period[2] | fp64 | 0.01687231554181534 | 0.01687231554181534 | 0.00e+00 | identical |
| 4.lab_o2_mass_err | fp64 | 0.0 | 0.0 | n/a | both negligible |
| 4.lab_o2_n | fp64 | 200.0 | 200.0 | 0.00e+00 | identical |
| 4.lab_o2_period_s | fp64 | 4.485701465466374 | 4.485701465466374 | 0.00e+00 | identical |
| 4.lab_o2_relL2_per_period[1] | fp64 | 0.09252213466431447 | 0.09252213466431447 | 0.00e+00 | identical |
| 4.lab_o2_relL2_per_period[2] | fp64 | 0.13984026463023488 | 0.13984026463023488 | 0.00e+00 | identical |
| 4.passed | fp64 | 1.0 | 1.0 | 0.00e+00 | identical |
| 5.passed | fp64 | 1.0 | 1.0 | 0.00e+00 | identical |
| 6.default_order | fp64 | 2.0 | 2.0 | 0.00e+00 | identical |
| 6.o1_mass_err | fp64 | 1.512885983111582e-16 | 2.4206175729785317e-16 | n/a | both negligible |
| 6.o2_mass_err | fp64 | 1.2571604866899507e-15 | 1.493802460655118e-15 | n/a | both negligible |
| 6.passed | fp64 | 1.0 | 1.0 | 0.00e+00 | identical |
| 6.t_equilibrium_s | fp64 | 478.3224327444235 | 478.3224327444235 | 0.00e+00 | identical |
| 7.event_mass_err | fp64 | 6.897042956748e-16 | 8.440437184831465e-16 | n/a | both negligible |
| 7.exchange_only_worst_rel | fp64 | 7.962285051070764e-14 | 7.962285051070764e-14 | n/a | both negligible |
| 7.passed | fp64 | 1.0 | 1.0 | 0.00e+00 | identical |
| 8.fp32_clip_m3 | fp32 | 0.0 | 0.0 | n/a | both negligible |
| 8.fp32_steps | fp32 | 40994.0 | 40994.0 | 0.00e+00 | identical |
| 8.fp64_clip_m3 | fp64 | 0.0 | 0.0 | n/a | both negligible |
| 8.fp64_mass_err | fp64 | 3.276340359397308e-15 | 3.4487793256813787e-15 | n/a | both negligible |
| 8.fp64_steps | fp64 | 40995.0 | 40995.0 | 0.00e+00 | identical |
| 8.passed | fp64 | 1.0 | 1.0 | 0.00e+00 | identical |

**Read this before trusting the verdict.** 2 double-precision metric(s) differ by more than the fp64 tolerance and are excused only because a perturbation that changes nothing physical moves them at least as far on a single machine:

- `4.field_o2_relL2_per_period[1]`: differs by 3.99e-03, own measured noise floor 1.58e-02
- `4.field_o2_relL2_per_period[2]`: differs by 8.58e-03, own measured noise floor 9.00e-03

That makes those metrics useless as corruption detectors -- not evidence that nothing is wrong. The verdict rests on the metrics that are well conditioned: the convergence orders, the mass errors, the equilibrium errors, and the step counts, which are integer and identical.

The suite agrees, so the local dataset and model were probably not corrupted -- but that is an argument about this suite's arrays, not a proof about every array in a 6-hour generation run. The dataset is regenerated here regardless; the surrogate's failure to learn effects rests on reasoning that this cross-check supports rather than replaces.
