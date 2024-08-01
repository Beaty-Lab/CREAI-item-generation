import itertools
import random
from config import config
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from statistics import mean


def SelectItemGenShots(
    itemPool: str,
    shotSelectionMetric: str,
    itemGenNumShots: int,
    round: int,
    shotSelectionSort: str,
    shotSelectionAggregate: str,
    seed: int,
    # if not using constraint satisfaction, defaults to a greedy method that just selects items based on originality scores
    shotSelectionAlgorithm: str,
):
    if shotSelectionAlgorithm == "constraint satisfaction":
        return ConstraintSatisfaction(
            itemPool,
            shotSelectionMetric,
            itemGenNumShots,
            round,
            shotSelectionSort,
            shotSelectionAggregate,
            seed,
        )
    elif shotSelectionAlgorithm == "greedy":
        itemPool = pd.read_json(f"{itemPool}_round_{round}.json", orient="records")
        # use greedy item selection
        if shotSelectionAggregate == "mean":
            meanItemScores = itemPool.groupby(f"creative_scenario_round_{round}").mean(
                numeric_only=True
            )
        elif shotSelectionAggregate == "var":
            meanItemScores = itemPool.groupby(f"creative_scenario_round_{round}").var(
                numeric_only=True
            )

        if shotSelectionSort == "max":
            meanItemScores.sort_values(
                by=f"{shotSelectionMetric}_round_{round}", ascending=False, inplace=True
            )
        elif shotSelectionSort == "min":
            meanItemScores.sort_values(
                by=f"{shotSelectionMetric}_round_{round}", ascending=True, inplace=True
            )

        item_list = meanItemScores.iloc[:itemGenNumShots].index.to_list()
        return item_list
    elif shotSelectionAlgorithm == "random":
        # TODO: implement random shot selection baseline
        itemPool = pd.read_json(f"{itemPool}_round_{round}.json", orient="records")
        item_list = itemPool.sample(n=itemGenNumShots, random_state=seed)[
            f"creative_scenario_round_{round}"
        ].to_list()
        return item_list
    elif shotSelectionAlgorithm == "zero shot":
        return []


# debug constrained optimization of item gen selection using results from prior greedy runs
# will always use the greedy selection from the prior round to seed the select paramters
def ConstraintSatisfaction(
    itemPool: str,
    shotSelectionMetric: str,
    itemGenNumShots: int,
    round: int,
    shotSelectionSort: str,
    shotSelectionAggregate: str,
    seed: int,
    delta: float = 0.02,
    sim_gamma: float = 0.001,
    originality_gamma: float = 0.002,
):
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )  # TODO: make an arg for config

    itemPoolDf = pd.read_json(f"{itemPool}_round_{round}.json", orient="records")
    if round != 0:
        priorItemPool = pd.read_json(
            f"{itemPool}_round_{round-1}.json", orient="records"
        )

        if shotSelectionAggregate == "mean":
            priorMeanItemScores = priorItemPool.groupby(
                f"creative_scenario_round_{round-1}"
            ).mean(numeric_only=True)
        elif shotSelectionAggregate == "var":
            priorMeanItemScores = priorItemPool.groupby(
                f"creative_scenario_round_{round-1}"
            ).var(numeric_only=True)

        priorMeanItemScores.sort_values(
            by=f"{shotSelectionMetric}_round_{round-1}", ascending=False, inplace=True
        )
        prior_items = priorMeanItemScores.iloc[:itemGenNumShots]
        prior_originality = prior_items[
            f"{shotSelectionMetric}_round_{round-1}"
        ].values.mean()
        prior_item_list = priorMeanItemScores.iloc[:itemGenNumShots].index.to_list()

        prior_embeddings = embedding_model.encode(
            prior_item_list,
            convert_to_tensor=True,
        )
        prior_sims = util.cos_sim(prior_embeddings, prior_embeddings)
        # remove the diagonal and keep only the lower triangle since the matrix is symmetric
        prior_sims = prior_sims.tril().flatten()
        prior_sim_score = prior_sims[prior_sims != 0].cpu().numpy().mean()

    else:
        # default to minimum values for the first iteration
        prior_sim_score = 1.0
        prior_originality = -2.0

    if shotSelectionAggregate == "mean":
        MeanItemScores = itemPoolDf.groupby(f"creative_scenario_round_{round}").mean(
            numeric_only=True
        )
    elif shotSelectionAggregate == "var":
        MeanItemScores = itemPoolDf.groupby(f"creative_scenario_round_{round}").var(
            numeric_only=True
        )

    all_item_combs = list(
        itertools.combinations(MeanItemScores.index.to_list(), itemGenNumShots)
    )

    # every time we fail to locate an optimized item set
    # we increase the target sim score and decrease the target originality by a factor gamma
    # unlikely, but it may be that this process repeats until we hit the base case for round 0
    # if that happens, or if round 0 fails to locate a satisfying item set
    # we default to the greedy approach
    with open(config["logFile"], "a") as log:
        while True:
            random.Random(seed).shuffle(all_item_combs)
            cur_item_comb = pd.DataFrame(
                columns=["item_comb", "sim_score", "originality"]
            )
            cur_item_comb = cur_item_comb.astype("object")
            for item_comb in tqdm(all_item_combs, total=len(all_item_combs)):
                item_embeddings = embedding_model.encode(
                    item_comb, convert_to_tensor=True
                )
                sims = util.cos_sim(item_embeddings, item_embeddings)
                # remove the diagonal and keep only the lower triangle since the matrix is symmetric
                sims = sims.tril().flatten()

                sim_score = sims[sims != 0].cpu().numpy().mean()
                originality_scores = []
                for item in item_comb:
                    originality_scores.append(
                        MeanItemScores.loc[item][f"{shotSelectionMetric}_round_{round}"]
                    )

                originality = mean(originality_scores)
                items = pd.DataFrame(
                    {
                        "originality": originality,
                        "sim_score": sim_score,
                        "item_comb": None,
                    },
                    index=[0],
                    dtype="object",
                )
                items.at[0, "item_comb"] = list(item_comb)

                cur_item_comb = pd.concat((cur_item_comb, items))
                cur_item_comb.reset_index(drop=True, inplace=True)

                print(
                    f"# Sim: {sim_score} Prior Sim: {prior_sim_score} # Score: {originality} Prior Score: {prior_originality} #"
                )
                log.writelines(
                    f"# Sim: {sim_score} Prior Sim: {prior_sim_score} # Score: {originality} Prior Score: {prior_originality} #\n"
                )

                # if sim_score <= prior_sim_score and (
                #     originality >= prior_originality
                #     or ((prior_originality - originality) <= delta)
                # ):
            matching_items = cur_item_comb.loc[
                (cur_item_comb["sim_score"] <= prior_sim_score)
                & (
                    (cur_item_comb["originality"] >= prior_originality)
                    | (prior_originality - cur_item_comb["originality"] <= delta)
                )
            ]

            if len(matching_items) > 0:
                matching_items = matching_items.sort_values(
                    by=["originality"], ascending=False
                )
                sim_score = matching_items.iloc[0]["sim_score"]
                originality = matching_items.iloc[0]["originality"]
                print(
                    f"# Sim: {sim_score} Prior Sim: {prior_sim_score} # Score: {originality} Prior Score: {prior_originality} #"
                )
                log.writelines(
                    f"# Sim: {sim_score} Prior Sim: {prior_sim_score} # Score: {originality} Prior Score: {prior_originality} #\n"
                )
                return matching_items.iloc[0]["item_comb"]

            # unlikely, but there may not be an item set that satisfies the constraints
            # default to greedy approach
            if prior_sim_score >= 1.0 and prior_originality <= 2.0:
                print(
                    "Failed to locate item set satisfying constraints, defaulting to greedy selection."
                )
                log.writelines(
                    "Failed to locate item set satisfying constraints, defaulting to greedy selection.\n"
                )
                return SelectItemGenShots(
                    itemPool,
                    shotSelectionMetric,
                    itemGenNumShots,
                    round,
                    shotSelectionSort,
                    shotSelectionAggregate,
                    useConstraintSatisfaction=False,
                )
            else:
                if (prior_originality - originality) > delta:
                    prior_originality -= originality_gamma
                if sim_score >= prior_sim_score:
                    prior_sim_score += sim_gamma
