"""Prompt registry for the agent C0/C1 eval.

Each entry: natural-language prompt + comparison rule. Reference values come from
../gold_references.json (loaded by the scorer). The eval shot list is injected into
the agent namespace as EVAL_SHOTS / ONE_SHOT (both conditions).

Tolerances are 1% (1e-2) for physics scalars (agents round their output; genuine
errors miss by far more), exact for counts. Set prompts score by Jaccard >= 0.95.
"""

EVAL_SHOTS = [
    198873, 198877, 198878, 198879, 198880, 198881, 198882, 198883, 198884, 198885,
    198886, 198887, 198888, 198889, 198890, 198891, 198892, 198893, 198894, 198895,
    198896, 198897, 198898, 198899, 198908, 198909, 198910, 198911, 198912, 198913,
    198914, 198915, 198916, 198917, 198918, 198919, 198920, 198921, 198922, 198923,
    198925, 198926, 198927, 198928, 198929, 198930, 198931, 198932, 198933, 198934,
]
ONE_SHOT = 198877

OUTPUT_CONTRACT = (
    "\n\nWhen finished, assign your final result to a Python variable named "
    "`answer`, shaped as the question implies: a dict keyed by shot number (int) "
    "for per-shot results, a list of shot numbers for a shot list, or a single "
    "scalar/small dict for an aggregate."
)

# rule forms (consumed by scorer.py):
#   ("scalardict", tol)        per-shot scalar; exact key set, value within rel tol
#   ("set",)                   shot set, Jaccard >= 0.95
#   ("scalar", tol)            single scalar within rel tol (0 => exact)
#   ("setdict", tol)           filtered shot set (Jaccard>=0.95) + per-shot scalar
#   ("twofield", tol, [keys])  per-shot dict of named scalars; exact key set
#   ("setdict2", tol, [keys])  filtered set (Jaccard>=0.95) + per-shot named scalars
#   ("aggregate", key, tol)    pull `key` scalar from a dict-style answer/ref
#   ("aggregate_one", tol)     ref is {shot: count}; answer is that scalar
PROMPTS = {
    "P01": dict(rule=("scalardict", 1e-2),
        text="For the DIII-D shots in the list EVAL_SHOTS, compute the peak plasma "
             "current in megamperes (MA) for each shot."),
    "P02": dict(rule=("scalardict", 1e-2),
        text="For the shots in EVAL_SHOTS, compute the maximum stored energy (Wmhd) "
             "in megajoules (MJ) for each shot."),
    "P03": dict(rule=("scalardict", 1e-2),
        text="For the shots in EVAL_SHOTS, report the plasma current in MA at "
             "2000 ms for each shot."),
    "P04": dict(rule=("scalardict", 1e-2),
        text="For the shots in EVAL_SHOTS, compute how long (in ms) the plasma "
             "current stayed above 1 MA in each shot."),
    "P05": dict(rule=("setdict", 1e-2),
        text="Among the first 21 shots in EVAL_SHOTS, find the maximum normalized "
             "beta in each shot and return the shots whose maximum normalized beta "
             "exceeded 2, with their peak values."),
    "P06": dict(rule=("scalardict", 1e-2),
        text="For the shots in EVAL_SHOTS, compute the peak total injected "
             "neutral-beam power in MW for each shot (report 0 MW for a shot "
             "that had no neutral beams)."),
    "P10": dict(rule=("set",),
        text="Find all plasma shots whose entry timestamp is on or after "
             "2024-06-01 and before 2024-06-08 (i.e. the calendar dates "
             "June 1 through June 7, 2024, inclusive)."),
    "P11": dict(rule=("aggregate", "n_plasma_june2024", 0),
        text="How many plasma shots were run in the calendar month June 2024 "
             "(entry timestamp on or after 2024-06-01 and before 2024-07-01)? "
             "Use the shot-type classification in the `shots_type` table "
             "(shot_type = 'plasma'). Assign the integer count to `answer`."),
    "P12": dict(rule=("setdict", 1e-2),
        text="Among plasma shots whose entry timestamp is on or after 2024-06-01 "
             "and before 2024-06-08 (calendar dates June 1 through June 7, 2024, "
             "inclusive), which reached a peak plasma current above 1.2 MA? Give "
             "those shots and their peak plasma current in MA."),
    "P13": dict(rule=("scalardict", 1e-2),
        text="Using the EFIT01 tree, get the peak plasma current (MA) from the node "
             "\\ipmhd for each shot in EVAL_SHOTS."),
    "P14": dict(rule=("twofield", 1e-2, ["max_bn", "peak_pnbi_MW"]),
        text="For the shots in EVAL_SHOTS, give the peak normalized beta and the "
             "peak total NBI power (MW) per shot (0 MW if the shot had no NBI). "
             "For each shot use a dict with keys 'max_bn' and 'peak_pnbi_MW'."),
    "P15": dict(rule=("setdict2", 1e-2, ["max_bn", "peak_pnbi_MW"]),
        text="Among the shots in EVAL_SHOTS, find those with peak normalized beta "
             "above 3 and give their peak normalized beta and peak total NBI "
             "power (MW). For each returned shot use a dict with keys 'max_bn' "
             "and 'peak_pnbi_MW'."),
    "P16": dict(rule=("scalardict", 1e-2),
        text="For the shots in EVAL_SHOTS, compute peak normalized beta divided by "
             "peak total NBI power (MW) per shot (only for shots with NBI)."),
    "P17": dict(rule=("scalardict", 5e-2),
        text="For the shots in EVAL_SHOTS, resample plasma current (Ip) and stored "
             "energy (Wmhd) onto a common uniform 10 ms time grid using "
             "zero-order-hold (pad / forward-fill) alignment, then report the "
             "Pearson correlation coefficient between them per shot."),
    "P19": dict(rule=("aggregate_one", 0),
        text="How many Thomson scattering electron-temperature channels are "
             "present for shot 198877?"),
    "P20": dict(rule=("aggregate_one", 0),
        text="For shot 198877, how many points describe the plasma boundary "
             "outline at its final equilibrium time slice?"),
    "P22": dict(rule=("aggregate", "mean_peak_ip_MA", 1e-2),
        text="For the shots in EVAL_SHOTS, count how many have valid plasma-current "
             "data and report their mean peak Ip (MA); ignore shots where Ip is "
             "unavailable. Assign a dict to `answer` with keys 'n_valid' (int) and "
             "'mean_peak_ip_MA' (float)."),
    "P23": dict(rule=("aggregate", "median_max_bn", 1e-2),
        text="What is the median peak normalized beta across the shots in "
             "EVAL_SHOTS? Assign the single median value (a float) to `answer`."),
}

def prompt_text(pid: str) -> str:
    return PROMPTS[pid]["text"] + OUTPUT_CONTRACT
