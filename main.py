import time

from quart import Quart
from quart import request, abort
from quart_cors import cors
from rouge import Rouge

rouge = Rouge()
app = Quart(__name__)
app = cors(app, allow_origin="*")


@app.route("/api/v1/example")
async def fetch_example():
    return {
        "error": 0,
        "data": {
            "article": "(CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency, Agencia Brasil. The American tourist died aboard the MS Veendam, owned by cruise operator Holland America. Federal Police told Agencia Brasil that forensic doctors were investigating her death. The ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension, according the agency. The other passengers came down with diarrhea prior to her death during an earlier part of the trip, the ship's doctors said. The Veendam left New York 36 days ago for a South America tour.",
            "gold_summary": "The elderly woman suffered from diabetes and hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says ."
        }
    }


def compute_summary(article):
    """
    TODO Call the model here
    :param article:
    :return:
    """
    return "The woman suffered from diabetes and hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says ."


@app.route("/api/v1/summary", methods=["POST"])
async def summarize():
    json = (await request.get_json())
    if 'article' not in json:
        return abort(400, description='Missing required field in json: article')
    t0 = time.perf_counter()
    the_summary = compute_summary(json['article'])

    return {
        "error": 0,
        "data": {
            "compute_summary": the_summary,
            "time": '{:.2f}'.format(1000 * (time.perf_counter() - t0))
        }
    }


def rate(gold_summary, compute_summary):
    r = rouge.get_scores(gold_summary, compute_summary)
    return {
        "error": 0,
        "data": {
            "rouge_1": r[0]['rouge-1']['f'],  # or p or r
            "rouge_2": r[0]['rouge-2']['f'],  # or p or r
            "rouge_l": r[0]['rouge-l']['f'],  # or p or r
        }
    }


@app.route("/api/v1/rate", methods=['POST'])
async def rate_the_summary():
    json = (await request.get_json())
    if 'gold_summary' not in json:
        return abort(400, description='Missing required field in json: gold_summary')
    if 'compute_summary' not in json:
        return abort(400, description='Missing required field in json: compute_summary')

    r = rate(json['gold_summary'], json['compute_summary'])

    return {
        "error": 0,
        "data": {
            "rouge_1": r[0],
            "rouge_2": r[1],
            "rouge_l": r[2],
        }

    }


if __name__ == "__main__":
    app.run(debug=True)
