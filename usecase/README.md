# Use cases

One directory here is one service. They share the library, the datasets under
`data/` and the checkpoints under `checkpoints/`; what they do not share is a
graph, a query or a model, because that is the part worth writing twice.

```
usecase/
  coldstart/app.py     recommend films from people, genres or titles you name
  genome/app.py        recommend films that belong beside a watchlist
  spotify/app.py       five songs to add to a handful you name
```

The two answer the same question from opposite ends, which is the point of
having both. `coldstart` has nothing to go on but a few names, so it *fits* an
affinity — TransD, a checkpoint, a warm-up. `genome` has the tag genome, where
the affinity is already measured, so it trains nothing and ranks with an
aggregate — `group_by(v.rec).agg(score=v.carried.score.sum())`. Same library,
same graph shape, different thing in the data.

## The convention

A directory is a use case when it holds an `app.py` exposing `app`. Nothing else
is required and nothing registers it: `run.sh` lists what is here, and the
image carries all of them.

```bash
./run.sh coldstart
```

That builds the image once, runs this one in a container named after it, and
mounts `data/` and `checkpoints/` from the working copy — so a fitted model
survives a restart and datasets are never baked into an image.

Paths inside a use case are relative to the repository root (`./data/movielens`,
`./checkpoints/ml.transd.npz`), which is the container's working directory and
also where you would run `uvicorn` by hand.

## The name is the address

The container takes the use case's name, so with a local DNS domain registered
it answers at `<usecase>.test` as well as on `localhost:8000`:

```bash
sudo container system dns create test     # once, per machine
./run.sh coldstart                        # -> http://coldstart.test:8000
```

One runs at a time: starting a use case stops whichever was serving, so the port
and the name both say what is up right now.

## Settings

A use case's settings are environment variables named after it, and `run.sh`
forwards exactly those into the container. The Spotify service reads the
100 000-playlist cut by default; the whole Million Playlist Dataset is one
variable away, plus the memory it needs:

```bash
SPOTIFY_DIR=./data/spotify/graph MEMORY=6g ./run.sh spotify
```

The graph is cached beside its files (`<dir>/.cache`). The first start on a
dataset builds that cache, which takes minutes on the whole graph. After that,
host and container both map the same cache in under a second, because it names
its files relative to itself. The factorization is still fitted at every start,
about four minutes on the whole graph. The service logs each startup step, so a
long start does not look like a hang.
