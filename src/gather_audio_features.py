import pandas as pd


def remove_indices(li, indices):
    c = li.copy()
    for i in sorted(indices, reverse=True):
        del c[i]

    return c


def num_of_times(num, div=100):
    times = 0
    while num > 0:
        num = num - div
        times = times + 1
    return times


def batchify(tids):
    times = num_of_times(len(tids))
    batchs = []
    for i in range(0, times):
        begin = 100 * i
        end = begin + 100
        if i == times - 1:
            batchs.append(tids[begin:])
        else:
            batchs.append(tids[begin:end])
    return batchs


def gather_playlist_audio_features(sp, playlist_id):
    offset = 0
    track_ids = []
    track_names = []
    artists = []
    track_urls = []
    source_pl = sp.playlist(playlist_id)['name']
    pl_names = []
    while True:
        response = sp.playlist_items(playlist_id,
                                     offset=offset,
                                     fields='items.track.id, items.track.artists, items.track.name, items.track.external_urls.spotify, total',
                                     additional_types=['track'])
        offset = offset + len(response['items'])

        if len(response['items']) == 0:
            break

        for i in response['items']:
            if i['track'] is None:
                continue
            track_ids.append(i['track']['id'])
            track_names.append(i['track']['name'])
            track_urls.append(i['track']['external_urls']['spotify'])
            pl_names.append(source_pl)
            a = ""
            for j in range(0, len(i['track']['artists'])):
                if j == 0:
                    a = i['track']['artists'][j]["name"]
                else:
                    a = a + ", " + i['track']['artists'][j]["name"]
            artists.append(a)

    batchs = batchify(track_ids)
    feats = sp.audio_features(batchs[0])

    nones = [i for i, v in enumerate(feats) if v is None]

    if len(nones) > 0:
        feats = remove_indices(feats, nones)
        track_names = remove_indices(track_names, nones)
        artists = remove_indices(artists, nones)
        track_urls = remove_indices(track_urls, nones)
        pl_names = remove_indices(pl_names, nones)

    track_names = batchify(track_names)
    artists = batchify(artists)
    track_urls = batchify(track_urls)
    pl_names = batchify(pl_names)

    df = pd.DataFrame(feats)
    df = df.drop(columns=['type', 'id', 'track_href', 'analysis_url'])
    df.insert(loc=0, column='track', value=track_names[0])
    df.insert(loc=1, column="artist", value=artists[0])
    df.insert(loc=2, column="url", value=track_urls[0])
    df.insert(loc=3, column="playlist", value=pl_names[0])

    for i in range(1, num_of_times(len(track_ids))):
        feats = sp.audio_features(batchs[i])
        df2 = pd.DataFrame(feats)
        df2 = df2.drop(columns=['type', 'id', 'track_href', 'analysis_url'])
        df2.insert(loc=0, column='track', value=track_names[i])
        df2.insert(loc=1, column="artist", value=artists[i])
        df2.insert(loc=2, column="url", value=track_urls[i])
        df2.insert(loc=3, column="playlist", value=pl_names[i])
        df = df.append(df2)

    return df
