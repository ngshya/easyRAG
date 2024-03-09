from wikipedia import page
from tqdm import tqdm
from re import split
from pandas import DataFrame, concat
from argparse import ArgumentParser


def wikipedia_page_to_df(title) -> DataFrame:
    p = page(title)
    list_chunks = []
    section = "Summary"
    for t in split('\n', p.content):
        if len(t) > 2:
            if t[:2] == "==":
                section = t.replace("=", "").strip()
            else:
                list_chunks.append({"section": section, "content": t.strip()})
    df_chunks = DataFrame(list_chunks)
    df_chunks["title"] = p.title
    df_chunks["url"] = p.url
    df_chunks = df_chunks.loc[:, ["title", "url", "section", "content"]]
    return df_chunks


if __name__ == "__main__":

    parser = ArgumentParser(
        prog='Fringe Wikipedia',
        description='Get Fringe data from Wikipedia'
    )

    parser.add_argument(
        '-l', '--links', 
        type=int, 
        default=-1,
        help='number of Wikipedia links to process'
    )

    args = parser.parse_args()

    list_links = [
        "Fringe (TV series)",
        "Pilot (Fringe)", 
        "The Same Old Story (Fringe)",
        "The Ghost Network (Fringe)", 
        "The Arrival (Fringe)", 
        "Power Hungry (Fringe)", 
        "The Cure (Fringe)", 
        "In Which We Meet Mr. Jones (Fringe)", 
        "The Equation (Fringe)", 
        "The Dreamscape (Fringe)", 
        "Safe (Fringe)", 
        "Bound (Fringe)", 
        "The No-Brainer (Fringe)", 
        "The Transformation (Fringe)", 
        "Ability (Fringe)", 
        "Inner Child (Fringe)", 
        "Unleashed (Fringe)", 
        "Bad Dreams (Fringe)", 
        "Midnight (Fringe)", 
        "The Road Not Taken (Fringe)", 
    ][:args.links]
    list_pages_df = []
    list_error = []
    pbar = tqdm(list_links)
    for l in pbar:
        pbar.set_description(l.ljust(30, ' ')[:30])
        try:
            list_pages_df.append(wikipedia_page_to_df(l))
        except Exception as e:
            list_error.append(str(e))
    df_pages = concat(list_pages_df)
    df_pages.reset_index(drop=True, inplace=True)
    df_pages.loc[:, ["content"]]\
        .to_csv("fringe_collection.tsv", sep="\t", index=True, header=False)
    df_pages.loc[:, ["title", "url", "section"]]\
        .to_csv("fringe_metadata.csv", sep=";", index=True, header=True)