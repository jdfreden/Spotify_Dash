from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from src.gather_audio_features import *


def log_reg(df, X_col, y, fit_int=True):
    X = df[X_col]
    model = LogisticRegression(fit_intercept=fit_int).fit(X, y)
    return model


def prediction_qual(df, X_col, y, fit_int=True, folds=5):
    df['y'] = y
    df_temp = df.sample(frac = 1)
    y = df_temp['y']
    X = df_temp[X_col]

    m = LogisticRegression(fit_intercept=fit_int)

    scores = cross_validate(m, X, y, cv=folds, scoring = ["balanced_accuracy", "recall", "precision"])
    print(scores)


def multi_predict_playlist(model, sp, playlist_ids, X_cols):
    df = gather_playlist_audio_features(sp, playlist_ids[0])

    for i in playlist_ids[1:]:
        df2 = gather_playlist_audio_features(sp, i)
        df = df.append(df2)

    probs = model.predict_proba(df[X_cols])
    like_prob = []
    for p in probs:
        like_prob.append(p[1])

    out = df.loc[:, ('track', 'artist')]
    out['prob'] = like_prob
    out['url'] = df['url']
    out['playlist'] = df['playlist']

    return out


def predict_playlist(model, sp, playlist_id, X_cols):
    df = gather_playlist_audio_features(sp, playlist_id)

    probs = model.predict_proba(df[X_cols])
    like_prob = []
    for p in probs:
        like_prob.append(p[1])

    out = df.loc[:, ('track', 'artist')]
    out['prob'] = like_prob
    out['url'] = df['url']
    out['playlist'] = df['playlist']

    return out


def prettify_predict_playlist(pp_out):
    cols = pp_out.columns
    cols = [x.capitalize() for x in cols]
    pp_out.columns = cols

    pp_out = pp_out.round({'Prob': 3})
    pp_out = pp_out.drop(columns = ['Url'])
    return pp_out
