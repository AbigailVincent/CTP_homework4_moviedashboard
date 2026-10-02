"""MovieLens 100K explorer: four questions, four charts.

Run with:  streamlit run app.py
Expects movie_ratings.csv in the same folder as this file.
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Relative to this file, so it works no matter where `streamlit run` is launched from.
DATA_PATH = Path(__file__).parent / "movie_ratings.csv"

st.set_page_config(page_title="Movie Ratings Explorer", layout="wide")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Load the ratings and turn the pipe-separated genres into a list column."""
    df = pd.read_csv(path)
    df["title"] = df["title"].str.strip()
    df["genre_list"] = df["genres"].str.split("|")
    return df


@st.cache_data
def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (rating, genre). A multi-genre movie appears once per genre."""
    return df.explode("genre_list").rename(columns={"genre_list": "genre"})


if not DATA_PATH.exists():
    st.error(f"Could not find {DATA_PATH.name} next to app.py. Put the CSV in the same folder.")
    st.stop()

ratings = load_data(DATA_PATH)
by_genre = explode_genres(ratings)
all_genres = sorted(by_genre["genre"].unique())

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
selected = st.sidebar.multiselect(
    "Genres",
    options=all_genres,
    default=[g for g in all_genres if g != "unknown"],
    help="Filters every chart. 'unknown' is a placeholder tag on only 2 movies, "
    "so it is off by default.",
)
floor_a = st.sidebar.slider("Minimum ratings per movie (Q4)", 1, 400, 50, step=5)
floor_b = st.sidebar.slider("Comparison floor (Q4)", 1, 400, 150, step=5)
min_year_ratings = st.sidebar.slider(
    "Minimum ratings per release year (Q3)",
    1, 500, 50, step=10,
    help="Release years with fewer ratings than this are hidden because their averages are noisy.",
)

if not selected:
    st.warning("Pick at least one genre in the sidebar.")
    st.stop()

sel = set(selected)
# Movies that carry at least one selected genre (used for Q3 and Q4).
in_selection = ratings[ratings["genre_list"].apply(lambda g: bool(sel.intersection(g)))]
genre_rows = by_genre[by_genre["genre"].isin(sel)]

st.title("Movie Ratings Explorer")
st.caption(
    f"{len(in_selection):,} ratings of {in_selection['movie_id'].nunique():,} movies "
    f"match your genre selection."
)

# ---------------------------------------------------------------- Q1
st.header("1. Genre breakdown")
st.markdown(
    "**How multi-genre movies are handled:** each movie's genre list is split, and the movie is "
    "counted once under *every* genre it carries. I count distinct movies (not individual "
    "ratings), so the bars add up to more than the number of movies."
)
q1 = (
    genre_rows.drop_duplicates(["movie_id", "genre"])
    .groupby("genre")["movie_id"].nunique()
    .rename("movies").reset_index()
)
st.altair_chart(
    alt.Chart(q1).mark_bar().encode(
        x=alt.X("movies:Q", title="Number of movies"),
        y=alt.Y("genre:N", sort="-x", title=None),
        tooltip=["genre", "movies"],
    ),
    use_container_width=True,
)

# ---------------------------------------------------------------- Q2
st.header("2. Genre satisfaction")
q2 = (
    genre_rows.groupby("genre")["rating"].agg(mean="mean", ratings="count").reset_index()
)
q2["mean"] = q2["mean"].round(3)
st.altair_chart(
    alt.Chart(q2).mark_bar().encode(
        x=alt.X("mean:Q", title="Mean rating", scale=alt.Scale(zero=False)),
        y=alt.Y("genre:N", sort="-x", title=None),
        tooltip=["genre", "mean", "ratings"],
    ),
    use_container_width=True,
)
top, bottom = q2.nlargest(3, "mean"), q2.nsmallest(3, "mean")
c1, c2 = st.columns(2)
c1.markdown("**Highest:** " + ", ".join(f"{r.genre} ({r.mean:.2f})" for r in top.itertuples()))
c2.markdown("**Lowest:** " + ", ".join(f"{r.genre} ({r.mean:.2f})" for r in bottom.itertuples()))
st.caption(
    "The bar axis does not start at 0, so differences look larger than they are. "
    "Small genres (e.g. Fantasy, Film-Noir) have fewer ratings behind their averages."
)

# ---------------------------------------------------------------- Q3
st.header("3. Ratings over time")
q3 = (
    in_selection.dropna(subset=["year"])
    .groupby("year")["rating"].agg(mean="mean", ratings="count").reset_index()
)
q3 = q3[q3["ratings"] >= min_year_ratings]
q3["year"] = q3["year"].astype(int)
q3["mean"] = q3["mean"].round(3)
if q3.empty:
    st.info("No release years meet the minimum. Lower the slider in the sidebar.")
else:
    st.altair_chart(
        alt.Chart(q3).mark_line(point=True).encode(
            x=alt.X("year:Q", title="Movie release year", axis=alt.Axis(format="d")),
            y=alt.Y("mean:Q", title="Mean rating", scale=alt.Scale(zero=False)),
            tooltip=["year", "mean", "ratings"],
        ),
        use_container_width=True,
    )
    st.caption(
        f"Release years with fewer than {min_year_ratings} ratings are hidden. "
        "Early years are bumpy because few ratings sit behind them."
    )

# ---------------------------------------------------------------- Q4
st.header("4. Best movies, with a floor")
movie_stats = (
    in_selection.groupby(["movie_id", "title"])["rating"]
    .agg(mean="mean", ratings="count").reset_index()
)


def top5(floor: int) -> pd.DataFrame:
    eligible = movie_stats[movie_stats["ratings"] >= floor]
    top = eligible.sort_values(["mean", "ratings"], ascending=False).head(5).copy()
    top["mean"] = top["mean"].round(3)
    return top


def show_top5(col, floor: int) -> pd.DataFrame:
    top = top5(floor)
    col.subheader(f"Minimum {floor} ratings")
    if top.empty:
        col.info("No movies meet this floor with the current genre selection.")
        return top
    col.altair_chart(
        alt.Chart(top).mark_bar().encode(
            x=alt.X("mean:Q", title="Mean rating", scale=alt.Scale(domain=[4, 4.6], clamp=True)),
            y=alt.Y("title:N", sort="-x", title=None),
            tooltip=["title", "mean", "ratings"],
        ),
        use_container_width=True,
    )
    col.dataframe(top[["title", "mean", "ratings"]], hide_index=True, use_container_width=True)
    return top


left, right = st.columns(2)
top_a = show_top5(left, floor_a)
top_b = show_top5(right, floor_b)

if not top_a.empty and not top_b.empty:
    a, b = set(top_a["title"]), set(top_b["title"])
    st.markdown(
        f"**What changes between {floor_a} and {floor_b}:** "
        f"dropped out: {', '.join(sorted(a - b)) or 'none'}; "
        f"moved in: {', '.join(sorted(b - a)) or 'none'}."
    )
st.caption("The bar axis is zoomed to 4.0–4.6 so small gaps are readable.")
