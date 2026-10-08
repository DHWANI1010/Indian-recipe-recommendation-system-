from flask import Flask, render_template, request

from recommender import DIETS, RecipeRecommender
rec = RecipeRecommender("IndianFoodDatasetCSV.csv")   # fitted once at startup

app = Flask(__name__)
rec = RecipeRecommender("IndianFoodDatasetCSV.csv")   # fitted once at startup

COURSES = ["Breakfast", "Lunch", "Dinner", "Snack", "Appetizer", "Side Dish",
           "Main Course", "Dessert"]


def _int(value):
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    form = {"ingredients": "", "diet": [], "exclude": "", "course": "", "cuisine": "",
            "max_time": "", "max_missing": "", "pantry_basics": False}
    recommendations, info = None, None

    if request.method == "POST":
        f = request.form
        form.update(
            ingredients=f.get("ingredients", "").strip(),
            diet=[d for d in f.getlist("diet") if d in DIETS],
            exclude=f.get("exclude", "").strip(),
            course=f.get("course", ""),
            cuisine=f.get("cuisine", "").strip(),
            max_time=f.get("max_time", ""),
            max_missing=f.get("max_missing", ""),
            pantry_basics=bool(f.get("pantry_basics")),
        )
        df, info = rec.recommend(
            form["ingredients"], diet=form["diet"], exclude=form["exclude"] or None,
            max_time=_int(form["max_time"]), course=form["course"] or None,
            cuisine=form["cuisine"] or None, max_missing=_int(form["max_missing"]),
            pantry_basics=form["pantry_basics"], top_k=12,
        )
        recommendations = df.to_dict("records")

    return render_template("index.html", form=form, diets=sorted(DIETS), courses=COURSES,
                           recommendations=recommendations, info=info)


if __name__ == "__main__":
    app.run(debug=True)
