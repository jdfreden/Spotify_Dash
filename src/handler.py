from src.logistic_reg import *
import spotipy


def handle(like_pl_id, dislike_pl_id, query):
    query = query.split(",")
    if len(query) == 1:
        multi = False
        pl_id = query[0]
    else:
        multi = True
        pl_id = query

    scope = "user-library-read"
    sp = spotipy.Spotify(auth_manager=spotipy.SpotifyOAuth(client_id="7eb0e8ee2ac24dbd8cc926961c21c2fe",
                                                           client_secret="fe795e0781ae479986c8c0a6560b25b5",
                                                           redirect_uri="http://localhost:8080",
                                                           scope=scope))

    # not including 'loudness' because it is correlated with energy
    X_cols = ["danceability", "energy", "speechiness", "acousticness", "liveness", "valence", "tempo"]
    df = gather_playlist_audio_features(sp, playlist_id=like_pl_id)
    df['like'] = [1] * len(df)

    df_not_like = gather_playlist_audio_features(sp, playlist_id=dislike_pl_id)
    df_not_like['like'] = [0] * len(df_not_like)

    df = df.append(df_not_like)
    y = df["like"]

    m = log_reg(df, X_cols, y, fit_int=False)

    if multi:
        rr = multi_predict_playlist(model=m, sp=sp, playlist_ids=pl_id, X_cols=X_cols)
    else:
        rr = predict_playlist(model=m, sp=sp, playlist_id=pl_id, X_cols=X_cols)

    rr = rr.sort_values(by=['prob'], ascending=False)

    rr = prettify_predict_playlist(rr)

    return rr

