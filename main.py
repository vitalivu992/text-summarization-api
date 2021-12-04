import time

from quart import Quart
from quart import request, abort
from quart_cors import cors
from rouge import Rouge

import inlinemodel

rouge = Rouge()
app = Quart(__name__)
app = cors(app, allow_origin="*")


@app.route("/api/v1/example")
async def fetch_example():
    try:
        example = inlinemodel.example_data()
        return {
            "error": 0,
            "data": {
                "article": example[0],
                "gold_summary": example[1]
            }
        }
    except Exception as e:
        return {
            "error": 1,
            "message": str(e)
        }


def compute_summary(article):
    try:
        return inlinemodel.summary(article)
    except Exception as e:
        return {
            "error": 1,
            "message": str(e)
        }


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
            "time": (time.perf_counter() - t0)
        }
    }


def rate(gold_summary, compute_summary):
    r = rouge.get_scores(gold_summary, compute_summary)
    metric = 'r'  # f or p or r
    return (r[0]['rouge-1'][metric],
            r[0]['rouge-2'][metric],
            r[0]['rouge-l'][metric])


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
    app.run(debug=True, host="0.0.0.0", port=5000)
