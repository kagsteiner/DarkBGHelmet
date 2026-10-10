# Running the engine on an Ubuntu VPS (no Docker)

The engine is one native binary, `bgserver`, plus a network file of about 4 MB. It runs under PM2
like your Node apps, listens only on `127.0.0.1`, and your Node app calls it over HTTP.

## Quick way: the build script

`deploy/builddarkbg.sh` does everything below in one go: clone or update, build, and start
or restart under PM2, on port 3050 by default. Copy it to your home folder and run it again
after every update. The sections below explain the individual steps.

## 1. Get the code onto the VPS

The repository is self-contained: the network is in `nets/latest.bgnn` and the
dependencies are in `third_party/`. Choose one way:

* **Through a private git remote (GitHub, GitLab, …).**
  * On the Mac: `git remote add origin <url>` and `git push -u origin master --tags`.
  * On the VPS: `git clone <url> ~/darkbghelmet`.
* **Copy it directly** from the Mac:

  ```
  rsync -a --exclude build --exclude data ./ user@vps:~/darkbghelmet/
  ```

## 2. Build

```
sudo apt update
sudo apt install -y build-essential cmake        # g++ 11+ (Ubuntu 22.04 or newer)
cd ~/darkbghelmet
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target bgserver bgcli
mkdir -p data
./build/bgcli move 4HPwATDgc/ABMA 31 2           # smoke test, also creates data/bearoff_os15.bin
```

The build optimises for the VPS's own CPU (`-march=native`). If you build on one machine and
run on another, configure with `-DBG_NATIVE=OFF`.

## 3. Run with PM2

```
pm2 start deploy/ecosystem.config.js
pm2 save                                         # restart automatically after a reboot
pm2 logs bgengine                                # should show "listening on http://127.0.0.1:8085"
curl -s localhost:8085/v1/info
```

Adjust `deploy/ecosystem.config.js` to the size of the VPS:

| Setting | Meaning |
|---|---|
| `--threads` | CPU cores one request may use. On a 2–4 core VPS, use 1–2 so your other apps stay responsive. |
| `--hash` | Memory for the search cache in MB. 64 is plenty for a quiz app. |
| `--plies` | Default depth. 2–3 answers in milliseconds; 4 can take up to about a second; 5+ is for offline analysis. |

After an update (`git pull`, then rebuild), run `pm2 restart bgengine`.

## 4. Calling it from Node

```js
async function bestMoves(xgid, dice, plies = 3) {
  const res = await fetch("http://127.0.0.1:8085/v1/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ position: { xgid }, dice, options: { plies, max_moves: 5 } }),
  });
  if (!res.ok) throw new Error((await res.json()).error);
  return res.json();   // { best, moves: [{ move, equity, diff, probabilities, ... }], time_ms }
}

async function cubeAction(xgid, plies = 3) {
  const res = await fetch("http://127.0.0.1:8085/v1/cube", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ position: { xgid }, options: { plies } }),
  });
  return res.json();   // { decision, take, equity: { no_double, double_take, double_pass }, ... }
}
```

The full request and response format is in `docs/API.md`. Positions can be given as XGID, as a
gnubg ID, or as a points array.

## Notes

* **Keep the port private.** Don't open 8085 in the firewall. If a browser ever needs direct
  access, put it behind your existing reverse proxy and your app's authentication. Deep searches
  are CPU‑expensive, so cap `plies` in your app.
* **Precompute quiz positions.** For a quiz, analyse the positions once at high depth
  (`plies` 4–5) and store the answers. The live server then only answers player-specific
  questions.
* **Concurrency.** The server handles one analysis at a time; others wait. At 2–3 ply each
  request takes only milliseconds.
