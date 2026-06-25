import pandas as pd

def explain(row):
    node = row["node"]
    pred = str(row["pred"])
    score = row["avg_edge_score"]  # FIXED

    # normalize class name safely
    pred_lower = pred.lower()

    if "athlete" in pred_lower:
        if score > 0.7:
            return (
                f"Node {node} is classified as ATHLETE because it has strong connections "
                f"to sports-related entities such as teams, competitions, and active players. "
                f"The graph structure shows strong participation in sports communities."
            )
        else:
            return (
                f"Node {node} is classified as ATHLETE due to moderate links with sports context, "
                f"but with weaker structural evidence."
            )

    elif "scientist" in pred_lower:
        if score > 0.7:
            return (
                f"Node {node} is classified as SCIENTIST because it is strongly connected to "
                f"academic institutions, research topics, and scholarly entities in the graph."
            )
        else:
            return (
                f"Node {node} is classified as SCIENTIST based on partial connections to "
                f"academic or research-related nodes."
            )

    else:
        return (
            f"Node {node} is classified as {pred} based on structural patterns in its neighborhood "
            f"with importance score {score:.4f}."
        )


def main():
    df = pd.read_csv("results/tables/grad_explanations.csv")

    df["explanation"] = df.apply(explain, axis=1)

    for e in df["explanation"].head(10):
        print(e)

    df.to_csv("results/tables/natural_language_explanations.csv", index=False)
    print("Saved: natural_language_explanations.csv")


if __name__ == "__main__":
    main()