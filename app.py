from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px

from utils.data_loader import fetch_price_history
from utils.news_analysis import get_daily_sentiment
from utils.features import prepare_dataset
from utils.modeling import (
    split_train_test,
    train_regressor,
    train_classifier,
    evaluate_regressor,
    evaluate_classifier,
    get_feature_importances,
)

# the template files live in the subdirectory "app/templates" rather
# than the default "templates" folder next to this script.  Specify the
# location explicitly so Flask can find them.
# create flask app and specify template and static directories
# templates live in app/templates, and we'll serve CSS/JS/images from app/static
app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static",
)


# symbol/display name pairs to show nicer labels in the dropdown
STOCK_LIST = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft Corp."),
    ("GOOGL", "Alphabet Inc."),
    ("AMZN", "Amazon.com Inc."),
    ("TSLA", "Tesla Inc."),
    ("META", "Meta Platforms Inc."),
    ("NVDA", "Nvidia Corp."),
    ("NFLX", "Netflix Inc."),
]


@app.route("/", methods=["GET"])
def index():
    context = {
        "stock_list": STOCK_LIST,
        "page_title": "Stock Market",
    }
    return render_template("index.html", **context)


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    context = {
        "stock_list": STOCK_LIST,
        "result": None,
        "chart": None,
        "metrics": None,
        "feature_importances": None,
        "explanation": {},
        "selected_ticker": None,
        "page_title": "Stock Dashboard",
    }

    if request.method == "POST":
        ticker = request.form.get("ticker", "").upper()
        context["selected_ticker"] = ticker

        # compute one‑year window ending today
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.DateOffset(years=1)

        hist = fetch_price_history(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        if hist.empty:
            context["result"] = "No price data available for that ticker/range."
        else:
            # pass same range to sentiment API
            daily_sent, positive, negative, _articles = get_daily_sentiment(ticker, start=start_date, end=end_date)
            X, y_reg, y_clf, hist2 = prepare_dataset(hist, daily_sent)

            (X_train, X_test,
             y_reg_train, y_reg_test,
             y_clf_train, y_clf_test) = split_train_test(X, y_reg, y_clf)

            scale = (
                len(y_clf_train[y_clf_train == 0]) /
                len(y_clf_train[y_clf_train == 1])
            ) if len(y_clf_train[y_clf_train == 1]) > 0 else 1

            reg_model = train_regressor(X_train, y_reg_train)
            clf_model = train_classifier(
                X_train, y_clf_train, scale_pos_weight=scale
            )

            context["metrics"] = {
                "reg": evaluate_regressor(reg_model, X_test, y_reg_test),
                "clf": evaluate_classifier(clf_model, X_test, y_clf_test),
            }
            fi = get_feature_importances(clf_model, X.columns.tolist())
            context["feature_importances"] = fi.head(5).to_dict("records")

            latest = X.iloc[-1:]
            next_price = reg_model.predict(latest)[0]
            prob_up = clf_model.predict_proba(latest)[0][1]

            context["explanation"]["next_price"] = next_price
            context["explanation"]["prob_up"] = prob_up
            context["explanation"]["rsi"] = hist2["RSI"].iloc[-1]
            context["explanation"]["latest_sentiment"] = hist2["Sentiment"].iloc[-1]

            context["positive_news"] = positive
            context["negative_news"] = negative

            fig = px.line(hist2, x="Date", y="Close", title=f"{ticker} Price Chart",
                          color_discrete_sequence=["#3b82f6"])
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#8ba3c4"),
                title=dict(font=dict(color="#f0f6ff", size=15)),
                xaxis=dict(gridcolor="rgba(99,179,237,0.08)", linecolor="rgba(99,179,237,0.15)"),
                yaxis=dict(gridcolor="rgba(99,179,237,0.08)", linecolor="rgba(99,179,237,0.15)"),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            context["chart"] = fig.to_html(full_html=False, config={"displayModeBar": False})

            # derive a simple text recommendation for the UI
            if prob_up > 0.7:
                rec_text, rec_cls = "BUY", "success"
            elif prob_up > 0.6:
                rec_text, rec_cls = "BUY", "primary"
            elif prob_up < 0.3:
                rec_text, rec_cls = "SELL", "danger"
            elif prob_up < 0.4:
                rec_text, rec_cls = "SELL", "warning"
            else:
                rec_text, rec_cls = "HOLD", "secondary"
            context["recommendation"] = {"text": rec_text, "class": rec_cls}

    return render_template("dashboard.html", **context)


@app.route("/sentiment", methods=["GET", "POST"])
def sentiment_page():
    # new page dedicated to news sentiment only
    context = {"stock_list": STOCK_LIST, "positive_news": [], "negative_news": [], "ticker": None, "trend_chart": None, "overall_chart": None, "articles": [], "page_title": "News Sentiment Analysis"}

    # accept ticker from POST form OR GET query param (e.g. redirected from Analyze)
    if request.method == "POST":
        ticker = request.form.get("ticker", "").upper()
    else:
        ticker = request.args.get("ticker", "").upper()

    if ticker:
        context["ticker"] = ticker
        # fixed one-year range
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.DateOffset(years=1)
        _, positive, negative, articles = get_daily_sentiment(ticker, start=start_date, end=end_date)
        context["positive_news"] = positive
        context["negative_news"] = negative
        context["articles"] = articles.to_dict("records")

        # compute trend and overall charts if we have article data
        if not articles.empty:
            # prepare counts per day and sentiment label
            counts = (
                articles.groupby(["Date", "label"]).size()
                .unstack(fill_value=0)
                .reset_index()
            )
            # ensure all sentiment columns exist
            for col in ["positive", "neutral", "negative"]:
                if col not in counts:
                    counts[col] = 0
            counts = counts.sort_values("Date")
            # take last 5 days if more rows
            if len(counts) > 5:
                counts = counts.tail(5)

            fig_trend = px.bar(counts,
                               x="Date",
                               y=["positive", "neutral", "negative"],
                               title="Sentiment Trend (Last 5 Days)",
                               color_discrete_map={
                                   "positive": "#10b981",
                                   "neutral":  "#8b5cf6",
                                   "negative": "#f43f5e",
                               })
            fig_trend.update_layout(
                barmode="stack",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#8ba3c4"),
                title=dict(font=dict(color="#f0f6ff", size=15)),
                xaxis=dict(gridcolor="rgba(99,179,237,0.08)", linecolor="rgba(99,179,237,0.15)"),
                yaxis=dict(gridcolor="rgba(99,179,237,0.08)", linecolor="rgba(99,179,237,0.15)"),
                legend=dict(font=dict(color="#8ba3c4")),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            context["trend_chart"] = fig_trend.to_html(full_html=False, config={"displayModeBar": False})

            overall = articles["label"].value_counts(normalize=True).reindex(["positive", "neutral", "negative"], fill_value=0)
            fig_overall = px.pie(values=overall.values,
                                 names=overall.index,
                                 title="Overall Sentiment",
                                 hole=0.4,
                                 color=overall.index,
                                 color_discrete_map={
                                     "positive": "#10b981",
                                     "neutral":  "#8b5cf6",
                                     "negative": "#f43f5e",
                                 })
            fig_overall.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#8ba3c4"),
                title=dict(font=dict(color="#f0f6ff", size=15)),
                legend=dict(font=dict(color="#8ba3c4")),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            context["overall_chart"] = fig_overall.to_html(full_html=False, config={"displayModeBar": False})

    return render_template("sentiment.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
