# Use cases

One directory here is one service. They share the library, the datasets under
`data/` and the checkpoints under `checkpoints/`; what they do not share is a
graph, a query or a model, because that is the part worth writing twice.

```
usecase/
  coldstart/app.py     recommend films from people, genres or titles you name
  genome/app.py        recommend films that belong beside a watchlist
```

The two answer the same question from opposite ends, which is the point of
having both. `coldstart` has nothing to go on but a few names, so it *fits* an
affinity — TransD, a checkpoint, a warm-up. `genome` has the tag genome, where
the affinity is already measured, so it trains nothing and ranks with one line —
`rank(Sum(carried.score))`. Same library, same graph shape, different thing in
the data.

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
