"""
BET THE GAP -- MA206 Poisson-process betting game (prototype)
Run:  streamlit run bet_the_gap.py
Deps: streamlit, numpy, matplotlib

Loop: cadets watch arrivals from a hidden-rate Poisson process, then bet
over/under on the NEXT gap against house lines that are deliberately
beatable by anyone who knows (a) median = ln2/lambda < mean, and
(b) memorylessness. Training mode explains each line after settling.
"""

import math
import random
import threading
import uuid

import matplotlib
matplotlib.use("Agg")  # headless backend; prevents blank figures on some hosts
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="Bet the Gap", page_icon="📻", layout="wide")

# ---------------- tuning knobs ----------------
START_BANKROLL = 100
WARMUP_GAPS = 12          # gaps shown before betting opens (rate-estimation data)
MEAN_GAP_RANGE = (20, 90) # hidden mean gap drawn uniformly from this (seconds)
P_DUE_LINE = 0.50         # chance of a gambler's-fallacy line after a long gap
P_MEAN_LINE = 0.55        # else: chance of a mean-anchored line (vs ~fair)
LONG_GAP_FACTOR = 1.5     # "long silence" = last gap > this * mean
ROUNDS = 20               # rounds per game -> comparable final scores for class

KIND_NAMES = {"mean_anchor": "Mean-anchored", "due": "\"Call is due\""}

LINE_EXPLANATIONS = {
    "mean_anchor": (
        "House anchored the line near the MEAN gap (1/\u03bb). But the "
        "exponential is right-skewed: the MEDIAN is ln2/\u03bb \u2248 0.69 of the "
        "mean, so most gaps come in UNDER a mean-anchored line. Correct side: UNDER."
    ),
    "due": (
        "After a long silence the house dropped the line, tempting the "
        "'a call is DUE' instinct. Memorylessness says the clock reset means "
        "nothing: the next gap is still Exp(\u03bb) and the fair line is still the "
        "median. A lowballed line makes OVER the value side."
    ),
    "fair": (
        "This line sat near the true median \u2014 a coin flip. Not every line is "
        "beatable; part of the skill is recognizing when there is no edge and "
        "betting small."
    ),
}


def new_game():
    mean_gap = random.uniform(*MEAN_GAP_RANGE)
    lam = 1.0 / mean_gap
    st.session_state.update(
        lam=lam,
        bankroll=START_BANKROLL,
        gaps=list(np.random.exponential(1 / lam, WARMUP_GAPS)),  # warmup history
        history=[],            # settled rounds
        line=None,             # pending line dict
        game_over=False,
    )


def post_line():
    """House posts an over/under line for the next gap. Exploitably biased."""
    lam = st.session_state.lam
    median, mean = math.log(2) / lam, 1 / lam
    last_gap = st.session_state.gaps[-1]
    if last_gap > LONG_GAP_FACTOR * mean and random.random() < P_DUE_LINE:
        kind, line = "due", median * random.uniform(0.55, 0.75)
    elif random.random() < P_MEAN_LINE:
        kind, line = "mean_anchor", mean * random.uniform(0.95, 1.15)
    else:
        kind, line = "fair", median * random.uniform(0.92, 1.08)
    line = max(1.5, math.floor(line) + 0.5)  # sportsbook half-point: no "ties"
    st.session_state.line = {"value": line, "kind": kind}


def settle(side: str, wager: int):
    lam = st.session_state.lam
    line = st.session_state.line
    gap = float(np.random.exponential(1 / lam))
    won = (side == "UNDER") == (gap < line["value"])
    st.session_state.bankroll += wager if won else -wager
    # the "correct" side a math-literate player should have taken:
    correct = "UNDER" if line["value"] > math.log(2) / lam else "OVER"
    st.session_state.history.append(
        dict(line=line["value"], kind=line["kind"], side=side, wager=wager,
             gap=gap, won=won, correct=correct, took_correct=(side == correct),
             bankroll=st.session_state.bankroll)
    )
    record_round(st.session_state.history[-1], lam)
    st.session_state.gaps.append(gap)
    st.session_state.line = None
    if st.session_state.bankroll <= 0 or len(st.session_state.history) >= ROUNDS:
        st.session_state.game_over = True
        t = class_tally()
        with t["lock"]:
            t["finals"].append(max(0, st.session_state.bankroll))
    update_board()


@st.cache_resource
def class_tally():
    """One tally shared by every browser connected to this server (the class)."""
    return dict(lock=threading.Lock(), players=set(), beatable=0, value_won=0,
                took_value=0, gaps=0, gaps_over_mean=0, finals=[],
                board={})  # leaderboard: callsign (lowercased) -> entry dict


def update_board():
    """Write this player's current game to the shared leaderboard."""
    s = st.session_state
    if not s.get("callsign"):
        return
    hist = s.history
    beatable = [h for h in hist if h["kind"] != "fair"]
    status = ("busted" if s.bankroll <= 0 else
              "finished" if s.game_over else "playing")
    t = class_tally()
    with t["lock"]:
        t["board"][s.callsign.lower()] = dict(
            pid=s.pid, name=s.callsign, bankroll=s.bankroll, round=len(hist),
            beatable=len(beatable), took_value=sum(h["took_correct"] for h in beatable),
            status=status, games=s.get("games", 1))


def board_rows():
    t = class_tally()
    with t["lock"]:
        entries = [dict(e) for e in t["board"].values()]
    entries.sort(key=lambda e: (-e["bankroll"], -e["round"]))
    return entries


def leaderboard(big=False):
    rows = board_rows()
    if not rows:
        st.write("No players yet. Enter a callsign to get on the board.")
        return
    me = st.session_state.get("callsign", "").lower()
    table = [{
        "Rank": i + 1,
        "Callsign": ("➤ " if e["name"].lower() == me and not big else "") + e["name"],
        "Bankroll": e["bankroll"],
        "Round": f"{e['round']}/{ROUNDS}",
        "Bet the value side": (f"{e['took_value'] / e['beatable']:.0%}"
                               if e["beatable"] else "—"),
        "Status": e["status"] + (f" (game {e['games']})" if e["games"] > 1 else ""),
    } for i, e in enumerate(rows)]
    st.dataframe(table, hide_index=True, use_container_width=True,
                 height=min(38 * (len(table) + 1) + 4, 900 if big else 420))
    st.caption("Ranked by bankroll. \"Bet the value side\" is how often a player "
               "took the math-favored side on beatable lines. Over 20 rounds luck "
               "can beat skill, but that column shows who played it right.")


def record_round(h, lam):
    t = class_tally()
    with t["lock"]:
        t["players"].add(st.session_state.pid)
        t["gaps"] += 1
        t["gaps_over_mean"] += h["gap"] > 1 / lam
        if h["kind"] != "fair":
            t["beatable"] += 1
            t["value_won"] += (h["gap"] < h["line"]) == (h["correct"] == "UNDER")
            t["took_value"] += h["took_correct"]


def class_panel():
    t = class_tally()
    with t["lock"]:
        snap = {k: (len(v) if isinstance(v, set) else
                    list(v) if isinstance(v, list) else v)
                for k, v in t.items() if k != "lock"}
    with st.expander(f"📣 Class results — {snap['players']} player(s), "
                     f"{snap['gaps']} bets so far", expanded=INSTRUCTOR):
        if not snap["gaps"]:
            st.write("No bets settled yet.")
            return
        c1, c2, c3 = st.columns(3)
        if snap["beatable"]:
            c1.metric("Value side won", f"{snap['value_won'] / snap['beatable']:.0%}",
                      help="Share of beatable lines where the math-favored side "
                           "won. Theory: about 60-68%.")
            c2.metric("Class took the value side",
                      f"{snap['took_value'] / snap['beatable']:.0%}",
                      help="How often players actually bet the math-favored side.")
        c3.metric("Gaps longer than the average", 
                  f"{snap['gaps_over_mean'] / snap['gaps']:.0%}",
                  help="Theory: e^-1 = 37%. Most gaps are SHORTER than the mean.")
        st.caption(f"Any one player's {ROUNDS} bets are mostly luck. Pooled over "
                   f"{snap['beatable']} beatable lines, the math shows through: "
                   f"the value side wins about 2 times in 3, and only about 37% "
                   f"of gaps outlast the average (the same e⁻¹ as the board).")
        if snap["finals"]:
            f = np.array(snap["finals"])
            st.write(f"Finished games: {len(f)} · median final bankroll "
                     f"**{np.median(f):.0f}** · best {f.max()} · "
                     f"{np.mean(f > START_BANKROLL):.0%} finished ahead of {START_BANKROLL}.")
    if INSTRUCTOR and st.button("🗑️ Reset class tally (instructor)"):
        with t["lock"]:
            t.update(players=set(), beatable=0, value_won=0, took_value=0,
                     gaps=0, gaps_over_mean=0, finals=[], board={})
        st.rerun()


def gap_distribution(g, training):
    """Histogram + density of every observed gap; grows by one gap each round."""
    top = float(g.max()) * 1.1
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=110)
    ax.hist(g, bins=np.histogram_bin_edges(g, bins="auto", range=(0, top)),
            density=True, color="tab:blue", alpha=0.35, edgecolor="white")
    # Gaussian KDE reflected at 0 so the curve doesn't leak into negative gaps
    x = np.linspace(0, top, 300)
    bw = max(1.06 * g.std() * len(g) ** -0.2, 1e-6)
    k = lambda u: np.exp(-0.5 * u ** 2) / math.sqrt(2 * math.pi)
    dens = (k((x[:, None] - g) / bw) + k((x[:, None] + g) / bw)).sum(1) / (len(g) * bw)
    ax.plot(x, dens, color="tab:blue", lw=2)
    ax.plot(g, np.zeros_like(g), "|", color="tab:blue", ms=10, alpha=0.6)
    ax.axvline(g[-1], color="tab:red", lw=1.5, label=f"newest gap ({g[-1]:.0f}s)")
    if training:
        ax.axvline(np.median(g), color="tab:green", ls="--",
                   label=f"median ({np.median(g):.0f}s)")
        ax.axvline(g.mean(), color="tab:orange", ls="--",
                   label=f"average ({g.mean():.0f}s)")
    ax.set_xlim(0, top); ax.set_yticks([]); ax.set_xlabel("gap between calls (s)")
    ax.set_title(f"Distribution of gaps — {len(g)} observed")
    ax.legend(frameon=False, fontsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)


# ---------------- app ----------------
# ?instructor=1  expands the class panel and shows the reset button
# ?view=board    projector view: leaderboard + class results, auto-refreshing
INSTRUCTOR = st.query_params.get("instructor") == "1"
if st.query_params.get("view") == "board":
    st.title("🏆 Bet the Gap — Leaderboard")

    @st.fragment(run_every="3s")
    def projector():
        leaderboard(big=True)
        class_panel()

    INSTRUCTOR = True  # the projector view always shows the class panel open
    projector()
    st.stop()

if "pid" not in st.session_state:
    st.session_state.pid = uuid.uuid4().hex
if "lam" not in st.session_state:
    new_game()

lam = st.session_state.lam
st.title("📻 Bet the Gap")

with st.expander("❓ How to play (read me first)", expanded=True):
    st.markdown(
        """
**The setup.** Radio calls arrive at a steady average rate — but *you don't know
the rate*. The timeline below shows every call so far. That history is your only
clue about how often calls tend to come.

**Each round:**
1. Press **Post the next line**. The house announces a number, like *"over/under
   60 seconds."*
2. You bet: will the **next gap** between calls be **longer** (OVER) or
   **shorter** (UNDER) than that number?
3. Choose a wager, pick a side, and the next call arrives. Right side = win your
   wager. Wrong side = lose it.

**The secret.** The house's lines are not always fair — they are set the way a
casino would set them to exploit common gut instincts about randomness. If you
apply what you know about waiting times (think: what's a *typical* gap versus an
*average* gap? does a long silence make a call "due"?), you can find the value
side and beat the book. If you bet on feel, the book beats you.

**Goal:** grow your bankroll. Bust at zero and it's over. Turn on **Training
mode** to get an explanation after every round; turn it off to test yourself.
        """
    )

# ---------------- callsign gate ----------------
if not st.session_state.get("callsign"):
    with st.form("join"):
        name = st.text_input("Your callsign (this is what the leaderboard shows)",
                             max_chars=20, placeholder="e.g. CDT Smith")
        joined = st.form_submit_button("Join the game", type="primary")
    if joined:
        name = " ".join(name.split())
        taken = class_tally()["board"].get(name.lower())
        if not name:
            st.error("Enter a callsign first.")
        elif taken and taken["pid"] != st.session_state.pid:
            st.error(f"\"{name}\" is already on the board. Pick a different callsign.")
        else:
            st.session_state.callsign = name
            st.session_state.games = 1
            update_board()
            st.rerun()
    st.stop()

left, right = st.columns([2, 1])

with left:
    # arrival-history strip: the cadet's rate-estimation data
    times = np.cumsum(st.session_state.gaps)
    if len(times):
        fig, ax = plt.subplots(figsize=(9, 1.6), dpi=110)
        ax.eventplot(times, colors="tab:blue", linelengths=0.8, linewidths=2)
        ax.set_xlim(0, float(times[-1]) * 1.02)
        ax.set_yticks([]); ax.set_xlabel("time (s)")
        ax.set_title(f"Call history \u2014 {len(times)} calls observed")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No call history yet \u2014 start a new game.")
    gaps = st.session_state.gaps
    listed = [f"{g:.0f}s" for g in gaps]
    if listed:
        listed[-1] = f"**{listed[-1]}**"  # newest gap, matching the strip's right end
    st.markdown(f"Gaps between calls, oldest → newest ({len(gaps)}): "
                + ", ".join(listed))
    st.caption("Use this history to judge the typical time between calls before you bet.")
    if len(gaps) >= 2:
        gap_distribution(np.array(gaps), st.session_state.get("training", True))

with right:
    ranking = [e["name"].lower() for e in board_rows()]
    me = st.session_state.callsign.lower()
    place = f"#{ranking.index(me) + 1} of {len(ranking)}" if me in ranking else None
    st.metric("Bankroll", st.session_state.bankroll, help="Leaderboard place: " + (place or "—"))
    st.caption(f"**{st.session_state.callsign}** · Round "
               f"{len(st.session_state.history)} of {ROUNDS}"
               + (f" · leaderboard {place}" if place else ""))
    training = st.toggle("Training mode (explain each line)", value=True,
                         key="training")
    if training:
        g = np.array(st.session_state.gaps)
        st.markdown(f"**Your data so far** ({len(g)} gaps)  \n"
                    f"average gap: **{g.mean():.0f}s**  \n"
                    f"median gap: **{np.median(g):.0f}s**")
        st.caption("Notice the median sits below the average. For any "
                   "exponential, median ≈ 0.69 × mean.")
    if st.button("🔄 New game (new hidden rate)"):
        new_game()
        st.session_state.games = st.session_state.get("games", 1) + 1
        update_board(); st.rerun()

st.divider()

if st.session_state.game_over:
    if st.session_state.bankroll <= 0:
        st.error("Busted. The house thanks you for your service. Start a new game.")
    else:
        st.success(f"Game over after {ROUNDS} rounds. Final bankroll: "
                   f"**{st.session_state.bankroll}** (started at {START_BANKROLL}).")
elif st.session_state.line is None:
    if st.button("📮 Post the next line", type="primary"):
        post_line(); st.rerun()
else:
    line = st.session_state.line
    st.subheader(f"House line: **{line['value']:.1f} seconds**")
    st.write(
        f"Will the next gap between calls be **longer** or **shorter** than "
        f"{line['value']:.1f} seconds? Pick a side and a wager."
    )
    # default must not exceed max, or Streamlit raises once bankroll < 10
    bank = max(1, st.session_state.bankroll)
    wager = st.number_input("Wager (points from your bankroll)", 1, bank,
                            min(10, bank))
    c1, c2 = st.columns(2)
    if c1.button(f"⬆️ OVER {line['value']:.1f}s", use_container_width=True):
        settle("OVER", int(wager)); st.rerun()
    if c2.button(f"⬇️ UNDER {line['value']:.1f}s", use_container_width=True):
        settle("UNDER", int(wager)); st.rerun()

# last-round result + pedagogy
if st.session_state.history:
    h = st.session_state.history[-1]
    msg = (f"Line {h['line']:.1f}s, gap was **{h['gap']:.1f}s** \u2192 you "
           f"{'WON' if h['won'] else 'LOST'} {h['wager']}.")
    (st.success if h["won"] else st.warning)(msg)
    if training:
        with st.expander("Why this line was (or wasn't) beatable", expanded=True):
            st.write(LINE_EXPLANATIONS[h["kind"]])
            if h["kind"] == "fair":
                st.write("No value side this round \u2014 the smart play was a "
                         "small wager.")
            else:
                verdict = "it \u2705" if h["took_correct"] else "the other side \u274c"
                st.write(f"The value side was **{h['correct']}** \u2014 you took "
                         f"{verdict}.")

# post-game debrief
if len(st.session_state.history) >= 5:
    with st.expander("📊 Debrief: your play vs the math"):
        st.markdown(
            "This panel reveals the hidden numbers and grades your *decisions*, "
            "not just your luck. **Taking the value side** means you bet the "
            "direction the math favored — you can take the value side and still "
            "lose a round (bad luck), or win on the wrong side (dumb luck). "
            "Beatable lines pay off only about 2 times in 3 \u2014 expect losing "
            "streaks even when you play perfectly. Over many rounds, value-side "
            "players win; over a few rounds, anything can happen. That is the "
            "difference between one bet and the average bet."
        )
        hist = st.session_state.history
        est = 1 / np.mean(st.session_state.gaps)
        st.write(f"Hidden truth: calls averaged one every **{1/lam:.0f}s**, but the "
                 f"*typical* (median) gap was only **{math.log(2)/lam:.0f}s** — "
                 f"most gaps run shorter than the average. Your observed history "
                 f"implied an average gap of about {1/est:.0f}s.")
        for kind in ("mean_anchor", "due"):
            rounds = [h for h in hist if h["kind"] == kind]
            if rounds:
                right_side = sum(h["took_correct"] for h in rounds)
                wins = sum(h["won"] for h in rounds)
                st.write(f"**{KIND_NAMES[kind]}** lines: took the value side "
                         f"{right_side}/{len(rounds)}, won {wins}/{len(rounds)}.")
        edge = [h["wager"] for h in hist if h["kind"] != "fair"]
        fair = [h["wager"] for h in hist if h["kind"] == "fair"]
        if edge and fair:
            st.write(f"**Bet sizing:** you wagered {np.mean(edge):.0f} on average "
                     f"when the line was beatable and {np.mean(fair):.0f} on "
                     f"coin-flip lines. Sharp bettors size up on an edge and "
                     f"down when there is none.")
        st.line_chart([START_BANKROLL] + [h["bankroll"] for h in hist])

with st.expander("🏆 Leaderboard", expanded=st.session_state.game_over):
    leaderboard()
class_panel()
