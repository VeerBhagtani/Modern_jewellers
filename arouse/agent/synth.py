"""Synthetic agent episodes for training and for Arouse AgentBench.

Every tool result in an episode comes from actually executing the gold action in the
Sandbox, so data and runtime always agree. The "test" split uses held-out tasks,
note facts, file names and sentence templates, so the benchmark measures
generalisation, not memorisation.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Callable

from arouse.agent.context import DAY_CODES, WEEKDAYS, build_context, fmt_dt, next_weekday_date
from arouse.agent.episode import Header
from arouse.agent.tools import REGISTRY, Reminder, Sandbox
from arouse.agent.wordlist import ADJECTIVES, NAMES, NOUNS, VERBS
from arouse.protocol import Action, Turn

# --- vocab pools (train / test disjoint) -------------------------------------------

TRAIN_TASKS = """check sales|call the vet|pay the electricity bill|order mineral blocks|update the milk records
water the plants|take my medicine|send the weekly report|clean the milking machine|review the invoices
book the vet visit|back up the sales file|call Ravi|renew the insurance|check the generator|study geography
buy groceries|call the feed supplier|pay the staff salaries|check the water tank|submit the tax form
email the bank|visit the market|fix the fence|order new liners|collect the milk payment|call mom
charge the tractor battery|clean the barn|count the stock|prepare the delivery list|feed the calves
check the cooling tank|go for a walk|read the contract|reply to Meena|pick up the parcel|service the tractor
refill the gas cylinder|update the price list|send the invoice to Green Mart|check the vaccination chart
buy lentils|prepare for the exam|call the plumber|clean the water trough|check the weather forecast
drink water|stretch|go to the gym|review the budget|order more feed|call the insurance agent
wash the milk cans|check the fodder stock|write the monthly report|pay the rent|book a doctor appointment
call the electrician|file the receipts|check emails|plan the week|buy vegetables|visit grandma
send the payment reminder|restock the shop|clean the kitchen|pay the phone bill|check the bank balance
call the transport company|renew the licence|order printer ink|backup my laptop|update the website
check the calf weights|order bedding straw|call the milk collection centre|record the milk yield
revise chemistry|finish the homework|practice the guitar|meditate|call the landlord|pay the water bill
collect the lab report|send the feed order|check the solar panels|sweep the yard|check the fuel level""".replace("\n", "|").split("|")

TEST_TASKS = """inspect the hay bales|call the accountant|renew the gym membership|check the tyre pressure
order ear tags|book the train tickets|return the library books|clean the air filter|call the seed dealer
water the vegetable garden|check the smoke alarm|pay the school fees|update the loan papers|call my brother
defrost the freezer|check the pregnancy test results for cow 12|send the photos to Priya|buy a birthday gift
repair the gate hinge|test the backup generator|call the vet about the lame cow|order spare pump parts
submit the subsidy application|clean the solar panels|check the rainfall gauge|pay the credit card bill
confirm the truck booking|feed the dogs|sharpen the knives|renew the passport""".replace("\n", "|").split("|")

TRAIN_FACTS = """the vet comes on Tuesday|feed delivery is delayed until Friday|Ravi is on leave next week
the cooling tank needs a new valve|milk price rises in October|cow 17 is eating less|the generator used 12 litres
Green Mart pays on the 5th|the fence near the pond is broken|we need 40 bags of feed|Meena handles the records
the tractor service is due|the bank changed the account number|order liners before Monday
the new calf was born today|water pressure was low this morning|Sharma Stores wants 20 more litres""".replace("\n", "|").split("|")
TEST_FACTS = """the hay supplier raised prices|cow 8 needs a hoof trim|the market is closed on Sunday
the pump makes a strange noise|Priya will visit on the 12th|the roof leaks in the calf shed""".replace("\n", "|").split("|")

# Compositional pools (train only): thousands of distinct tasks/facts/files so the model
# learns to COPY the user's words instead of memorising a fixed list.
PEOPLE = ["Ravi", "Meena", "Priya", "Arjun", "Sunita", "Rahul", "Anita", "Vikram", "Neha", "Kiran", "Deepak", "Pooja",
          "Suresh", "Lakshmi", "Amit", "Kavya", "Rohan", "Divya", "Manoj", "Asha", "Farhan", "Gita", "Harish", "Isha",
          "Jaya", "the vet", "the bank", "the supplier", "the landlord", "mom", "dad", "the doctor", "my boss",
          "the plumber", "the dairy", "the school", "the driver", "Mr Sharma", "Mrs Iyer", "the milkman"]
THINGS = ["the invoices", "the milk cans", "the feed order", "the tractor", "the water pump", "the report", "the barn",
          "the shop", "the documents", "the car", "the gas bill", "the loan papers", "the tickets", "the prescription",
          "the laptop", "the order", "the budget", "the cow shed", "the price list", "the delivery van", "the fridge",
          "the stock sheet", "the bank statement", "the salary slips", "the rain gutter", "the pressure washer"]
ITEMS = ["bags of feed", "litres of milk", "kg of rice", "eggs", "batteries", "bulbs", "notebooks", "liners",
         "bottles of ghee", "packets of curd", "bales of straw", "sacks of cement", "boxes of gloves"]
VERB_PERSON = ["call", "email", "text", "meet", "pay", "visit", "call back", "message", "thank", "invite"]
VERB_THING = ["check", "clean", "fix", "review", "renew", "pick up", "print", "sign", "submit", "update", "wash",
              "repair", "inspect", "send", "file", "sort", "photograph", "return", "prepare", "test"]
SYLLABLES = ["ka", "ren", "to", "vi", "lo", "mar", "sa", "den", "ru", "pel", "ti", "zo", "bar", "nu", "fel", "qui"]
FILE_STEMS = ["sales", "orders", "milk", "feed", "staff", "stock", "expenses", "invoices", "customers", "records",
              "report", "budget", "prices", "vendors", "animals", "yield", "fuel", "repairs", "salary", "notes", "todo"]
TEST_PHRASES = {t.lower() for t in TEST_TASKS} | {f.lower() for f in TEST_FACTS}


def pseudo_word(rng: random.Random) -> str:
    return "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize()


def compositional_task(rng: random.Random) -> str:
    k = rng.random()
    person = pseudo_word(rng) if rng.random() < 0.15 else rng.choice(PEOPLE)
    if k < 0.3:
        return f"{rng.choice(VERB_PERSON)} {person}"
    if k < 0.55:
        return f"{rng.choice(VERB_THING)} {rng.choice(THINGS)}"
    if k < 0.75:
        return f"{rng.choice(['buy', 'order', 'collect', 'deliver'])} {rng.randint(2, 60)} {rng.choice(ITEMS)}"
    if k < 0.9:
        return f"send {rng.choice(THINGS)} to {person}"
    return f"{rng.choice(VERB_THING)} {rng.choice(THINGS)} with {person}"


def wild_task(rng: random.Random) -> str:
    """Unpredictable but natural-looking phrase: only copying reproduces it."""
    name = pseudo_word(rng) if rng.random() < 0.3 else rng.choice(NAMES)
    adj = rng.choice(ADJECTIVES) + " " if rng.random() < 0.5 else ""
    noun, noun2 = rng.choice(NOUNS), rng.choice(NOUNS)
    v = rng.choice(VERBS)
    return rng.choice([
        f"{v} the {adj}{noun}", f"{v} the {noun} and the {noun2}", f"{v} {name}", f"{v} the {adj}{noun} for {name}",
        f"{v} {rng.randint(2, 99)} {noun}s", f"{v} {name}'s {noun}", f"{v} the {noun} at the {noun2}",
        f"{v} the {pseudo_word(rng).lower()} {noun}",
    ])


def wild_fact(rng: random.Random) -> str:
    name = pseudo_word(rng) if rng.random() < 0.3 else rng.choice(NAMES)
    adj = rng.choice(ADJECTIVES)
    return rng.choice([
        f"{name} will {rng.choice(VERBS)} the {adj} {rng.choice(NOUNS)}", f"the {rng.choice(NOUNS)} is {adj}",
        f"{name} needs {rng.randint(2, 99)} {rng.choice(NOUNS)}s", f"the {adj} {rng.choice(NOUNS)} is at the {rng.choice(NOUNS)}",
        f"{name} owes {rng.randint(10, 9999)} rupees for the {rng.choice(NOUNS)}",
    ])


def compositional_fact(rng: random.Random) -> str:
    person = pseudo_word(rng) if rng.random() < 0.15 else rng.choice(PEOPLE[:25])
    day = rng.choice(WEEKDAYS).capitalize()
    return rng.choice([
        f"{person} will visit on {day}", f"{person} is on leave until {day}",
        f"{rng.choice(ITEMS)} cost {rng.randint(5, 900)} rupees now", f"we used {rng.randint(2, 90)} {rng.choice(ITEMS)} this week",
        f"{rng.choice(THINGS)} needs repair", f"{person} owes {rng.randint(100, 9000)} rupees",
        f"the meeting with {person} moved to {day}",
    ])


def compositional_file(rng: random.Random) -> str:
    stem = rng.choice(FILE_STEMS)
    return rng.choice([f"{stem}.csv", f"{stem}.txt", f"{stem}_{rng.randint(2020, 2027)}.csv", f"{stem}_{rng.choice(MONTHS).lower()}.csv"])


TRAIN_FILES = ["sales.csv", "invoices.csv", "milk_records.csv", "notes.txt", "todo.txt", "staff.csv",
               "feed_orders.csv", "expenses.csv", "report.txt", "inventory.csv", "customers.csv"]
TEST_FILES = ["suppliers.csv", "payments.csv", "vet_visits.txt", "herd.csv", "deliveries.csv"]
CSV_HEADERS = ["date,product,quantity,price", "date,customer,amount", "date,cow,litres", "name,role,phone",
               "item,count", "date,item,cost"]

GREETINGS = ["hi", "hello", "hey", "hey Arouse", "good morning", "good evening", "hello there", "hi Arouse", "namaste"]
TEST_GREETINGS = ["yo", "hiya", "good afternoon"]
THANKS = ["thanks", "thank you", "thanks a lot", "great, thanks", "thank you so much", "perfect, thanks", "ok thanks"]
HELP = ["what can you do?", "help", "how can you help me?", "what are you able to do?", "what do you do?"]
TEST_HELP = ["tell me what you can do"]
WHO = ["who are you?", "what is your name?", "are you a bot?", "what are you?"]
OUT_OF_SCOPE = ["what is the capital of France?", "tell me a joke", "write a poem about rain", "what is 17 times 23?",
                "who won the cricket match yesterday?", "translate hello into Spanish", "what's the weather today?",
                "explain quantum physics", "write a python script to sort a list", "what is the meaning of life?",
                "recommend a movie", "how tall is Mount Everest?", "what's the latest news?", "sing me a song",
                "how do I cook biryani?", "what is the price of gold?", "who is the prime minister?"]
TEST_OUT_OF_SCOPE = ["what's the population of Mumbai?", "tell me a story", "solve 2x + 3 = 11",
                     "what's the stock price of Tata Motors?"]
# --- v3: extra train-only variety (held-out test phrasings and entities stay excluded) ---
HOW_ARE_YOU = ["how are you?", "how are you doing?", "how's it going?", "what's up?"]
BYE = ["bye", "goodbye", "see you", "good night", "bye bye", "see you later"]
ACKS = ["ok", "okay", "cool", "great", "nice", "alright", "got it"]
_OOS_PLACES = ["France", "Japan", "Delhi", "Kerala", "Brazil", "Canada", "Egypt", "London", "Pune", "Nepal", "Kenya",
               "Paris", "Chennai", "Australia", "Germany", "Goa", "Russia", "Peru"]
_OOS_ATTR = ["capital", "population", "currency", "area", "language", "president", "climate", "history"]
_OOS_KINDS = ["joke", "poem", "fun fact", "riddle", "song", "limerick", "quote"]
_OOS_TOPICS = ["rain", "cows", "the moon", "friendship", "cricket", "the sea", "trains", "mountains", "coffee", "space"]
_OOS_EXPLAIN = ["quantum physics", "black holes", "photosynthesis", "inflation", "machine learning", "gravity",
                "the stock market", "democracy", "electricity", "evolution"]
_OOS_HOW = ["cook biryani", "learn python", "fix a flat tyre", "lose weight", "play chess", "bake bread", "invest money",
            "write a resume", "grow tomatoes", "tie a tie"]
_OOS_THINGS = ["gold", "petrol", "bitcoin", "onions", "silver", "the dollar", "diesel", "tomatoes"]
_OOS_LANGS = ["Spanish", "French", "Hindi", "German", "Tamil", "Japanese"]
_OOS_WORDS = ["hello", "thank you", "good morning", "water", "friend", "milk"]


def oos_question(rng: random.Random) -> str:
    """A general request Arouse cannot do (it has no world knowledge): must be refused, not routed to a tool."""
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    return rng.choice([
        f"what is the {rng.choice(_OOS_ATTR)} of {rng.choice(_OOS_PLACES)}?",
        f"tell me a {rng.choice(_OOS_KINDS)}", f"tell me a {rng.choice(_OOS_KINDS)} about {rng.choice(_OOS_TOPICS)}",
        f"write a {rng.choice(_OOS_KINDS)} about {rng.choice(_OOS_TOPICS)}",
        f"what is {a} {rng.choice(['times', 'plus', 'minus', 'divided by'])} {b}?", f"calculate {a} * {b}",
        f"solve {a}x - {b} = {rng.randint(1, 50)}", f"translate {rng.choice(_OOS_WORDS)} into {rng.choice(_OOS_LANGS)}",
        f"explain {rng.choice(_OOS_EXPLAIN)}", f"how do I {rng.choice(_OOS_HOW)}?", f"what is the price of {rng.choice(_OOS_THINGS)}?",
        f"what's the weather in {rng.choice(_OOS_PLACES)}?", f"who won the {rng.choice(['football', 'cricket', 'tennis'])} match?",
        f"recommend a {rng.choice(['movie', 'book', 'song', 'restaurant', 'game'])}", f"what do you think about {rng.choice(_OOS_TOPICS)}?",
        f"how far is {rng.choice(_OOS_PLACES)} from {rng.choice(_OOS_PLACES)}?", f"define {rng.choice(_OOS_EXPLAIN)}",
        f"write code to {rng.choice(['sort a list', 'reverse a string', 'parse a CSV', 'build a website'])}",
        f"what's the news about {rng.choice(_OOS_PLACES)}?", f"who invented {rng.choice(['the telephone', 'paper', 'the radio'])}?",
    ])


AUG_PREFIX = ["", "", "", "please ", "hey, ", "hi Arouse, ", "can you ", "Arouse, ", "quick one: ", "ok so "]
AUG_SUFFIX = ["", "", "", " please", " thanks", " thank you", "!", " pls"]

YES = ["yes", "yes please", "sure", "yeah", "ok, do that", "yes, do it"]
NO = ["no", "no thanks", "never mind", "no, leave it"]

CAPABILITIES = "I can set reminders, list or cancel them, take notes and read your files."
R_GREET = "Hello! " + CAPABILITIES + " What would you like to do?"
R_THANKS = "You're welcome!"
R_HELP = CAPABILITIES + " For example: 'Remind me tomorrow at 8 AM to check sales.'"
R_WHO = "I'm Arouse, a task assistant. " + CAPABILITIES
R_OUT = "I can't help with that yet. " + CAPABILITIES
R_HOW = "I'm doing well, thanks! " + CAPABILITIES
R_BYE = "Goodbye!"
R_ACK = "Okay! Anything else?"

Q_TIME = "What time should I remind you?"
Q_WHEN = "When should I remind you?"
Q_TASK = "What should I remind you about?"

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def hm(h: int, m: int) -> str:
    return f"{h:02d}:{m:02d}"


def say_time(rng: random.Random, h: int, m: int) -> str:
    h12, ampm = (h % 12 or 12), ("AM" if h < 12 else "PM")
    base = f"{h12}" if m == 0 else f"{h12}:{m:02d}"
    forms = [f"{base} {ampm}", f"{base} {ampm.lower()}", f"{base}{ampm.lower()}", hm(h, m)]
    if h < 12:
        forms.append(f"{base} in the morning")
    elif h < 17:
        forms.append(f"{base} in the afternoon")
    elif h < 21:
        forms.append(f"{base} in the evening")
    else:
        forms.append(f"{base} at night")
    if (h, m) == (12, 0):
        forms.append("noon")
    return rng.choice(forms)


def rule_words(repeat: dict[str, Any]) -> str:
    f = repeat["freq"]
    if f == "daily":
        n = repeat.get("interval", 1)
        return "every day" if n == 1 else f"every {n} days"
    if f == "monthly":
        return f"on the {ordinal(repeat['by_month_day'][0])} of every month"
    days = repeat["by_day"]
    if days == list(DAY_CODES[:5]):
        return "every weekday"
    if days == ["SA", "SU"]:
        return "every weekend"
    names = [WEEKDAYS[DAY_CODES.index(d)].capitalize() for d in days]
    return "every " + " and ".join(names)


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def created_msg(res: dict[str, Any], repeat: dict[str, Any] | None) -> str:
    date, time = res["next_run"].split("T")
    msg = f"Reminder set: {res['task']} on {date} at {time}"
    return msg + (f", repeating {rule_words(repeat)}." if repeat else ".")


# --- episode builder -------------------------------------------------------------------


class Builder:
    def __init__(self, now: datetime, files: dict[str, str], reminders: list[Reminder], failures: dict[str, int]):
        self.now = now
        self.init = {"files": dict(files), "reminders": [r.to_dict() for r in reminders], "failures": dict(failures)}
        self.sandbox = Sandbox(now, files, reminders, failures)
        self.events: list[dict[str, Any]] = []
        self.skills: list[list[str]] = []  # per arouse event

    def user(self, text: str) -> None:
        self.events.append({"type": "user", "content": text})

    def act(self, action: Action, skills: list[str], plan: str | None = None, verify: str | None = None):
        self.events.append({"type": "arouse", "turn": Turn(action, plan, verify).to_dict()})
        self.skills.append(sorted(set(skills + ["structured_output"])))
        if action.type != "tool_call":
            return None
        ok, payload = self.sandbox.execute(action.tool, action.arguments)
        self.events.append({"type": "tool_result" if ok else "tool_error", "content": payload})
        return ok, payload

    def header(self) -> Header:
        return Header(context=build_context(self.now), tools=REGISTRY.names())

    def to_dict(self, eid: str, category: str, split: str) -> dict[str, Any]:
        return {"id": eid, "split": split, "category": category, "now": fmt_dt(self.now), **self.init,
                "events": self.events, "skills": self.skills, "final_state": self.sandbox.snapshot()}


def episode_header(ep: dict[str, Any]) -> Header:
    now = datetime.strptime(ep["now"], "%Y-%m-%dT%H:%M")
    return Header(context=build_context(now), tools=REGISTRY.names())


def episode_sandbox(ep: dict[str, Any]) -> Sandbox:
    now = datetime.strptime(ep["now"], "%Y-%m-%dT%H:%M")
    rems = [Reminder(r["task_id"], r["task"], datetime.strptime(r["next_run"], "%Y-%m-%dT%H:%M"), r.get("repeat"))
            for r in ep["reminders"]]
    return Sandbox(now, ep["files"], rems, ep["failures"])


class Generator:
    def __init__(self, seed: int, split: str = "train") -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        self.rng = random.Random(seed)
        self.split = split
        test = split == "test"
        self.tasks = TEST_TASKS if test else TRAIN_TASKS
        self.facts = TEST_FACTS if test else TRAIN_FACTS
        self.file_pool = TEST_FILES + TRAIN_FILES[:4] if test else TRAIN_FILES
        self.test = test
        self.greetings = TEST_GREETINGS if test else GREETINGS
        self.help = TEST_HELP + HELP[:1] if test else HELP
        self.oos = TEST_OUT_OF_SCOPE if test else OUT_OF_SCOPE
        self.aug = random.Random(seed * 7919 + 17)  # separate stream: never perturbs the main one

    def task(self) -> str:
        """Train: 30% fixed phrases, 30% compositional, 40% wild (forces copying)."""
        k = self.rng.random()
        if self.test or k < 0.3:
            return self.rng.choice(self.tasks)
        while True:
            t = compositional_task(self.rng) if k < 0.6 else wild_task(self.rng)
            if t.lower() not in TEST_PHRASES:
                return t

    def fact(self) -> str:
        k = self.rng.random()
        if self.test or k < 0.3:
            return self.rng.choice(self.facts)
        while True:
            f = compositional_fact(self.rng) if k < 0.6 else wild_fact(self.rng)
            if f.lower() not in TEST_PHRASES:
                return f

    def file_names(self, k: int) -> list[str]:
        if self.test:
            return self.rng.sample(self.file_pool, k)
        names: set[str] = set()
        while len(names) < k:
            f = self.rng.choice(self.file_pool) if self.rng.random() < 0.5 else compositional_file(self.rng)
            if f not in TEST_FILES:
                names.add(f)
        return sorted(names)

    def say(self, text: str, chat: bool = False) -> None:
        """Add a user message; in train, sometimes wrap it in a polite prefix/suffix (chit-chat: suffix only)."""
        if not self.test and self.aug.random() < 0.5:
            pre = "" if chat else self.aug.choice(AUG_PREFIX)
            suf = self.aug.choice(["", "!", " :)"] if chat else AUG_SUFFIX)
            body = text[:1].lower() + text[1:] if pre else text
            if suf and body[-1:] in ".!?":
                body = body[:-1]
            text = pre + body + suf
        self._b.user(text)

    def pick(self, train: list[str], test: list[str]) -> str:
        """Sentence template: test split uses only the held-out templates."""
        return self.rng.choice(test if self.split == "test" else train)

    # --- random world -----------------------------------------------------------------

    def world(self, failures: dict[str, int] | None = None) -> Builder:
        r = self.rng
        now = datetime(2026, 1, 1, 6, 0) + timedelta(minutes=5 * r.randrange(0, 12 * 24 * 730))
        now = now.replace(hour=r.randrange(6, 23))
        files = {}
        for name in self.file_names(r.randint(2, 5)):
            rows = [r.choice(CSV_HEADERS)] + [f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d},{r.randint(1, 99)}"
                                               for _ in range(r.randint(1, 12))]
            files[name] = "\n".join(rows)
        rems = []
        for i, task in enumerate(dict.fromkeys(self.task() for _ in range(r.randint(0, 3)))):
            when = now + timedelta(minutes=30 * r.randint(1, 400))
            rep = r.choice([None, None, {"freq": "weekly", "by_day": [r.choice(DAY_CODES)]}, {"freq": "daily"}])
            rems.append(Reminder(f"r-{i + 1}", task, when.replace(second=0), rep))
        return Builder(now, files, rems, failures or {})

    def time(self) -> tuple[int, int]:
        return self.rng.randint(5, 22), self.rng.choice([0, 0, 0, 0, 15, 30, 30, 45, 10, 20, 40, 5, 50])

    def day(self, b: Builder) -> tuple[str, str, str]:
        """(phrase, resolved YYYY-MM-DD, label used in the plan's lookup step)."""
        r, now = self.rng, b.now
        kind = r.choices(["tomorrow", "weekday", "date"], [4, 5, 2])[0]
        if kind == "tomorrow":
            return "tomorrow", (now + timedelta(days=1)).strftime("%Y-%m-%d"), "Tomorrow"
        if kind == "weekday":
            w = r.randrange(7)
            name = WEEKDAYS[w].capitalize()
            phrase = r.choice([f"on {name}", f"this {name}", f"next {name}", f"on {name.lower()}"])
            return phrase, next_weekday_date(now, w).strftime("%Y-%m-%d"), name
        d = now + timedelta(days=r.randint(2, 75))
        m = MONTHS[d.month - 1]
        phrase = r.choice([f"on {m} {d.day}", f"on {d.day} {m}", f"on {m[:3]} {d.day}", f"on the {ordinal(d.day)} of {m}"])
        return phrase, d.strftime("%Y-%m-%d"), f"{m} {d.day}"

    def rule(self) -> tuple[str, dict[str, Any], int | None]:
        """(phrase, repeat, forced hour or None)."""
        r = self.rng
        kind = r.choices(["weekly", "pair", "weekdays", "weekend", "daily", "every_n", "monthly"], [5, 2, 2, 1, 3, 2, 2])[0]
        if kind == "weekly":
            w = r.randrange(7)
            name = WEEKDAYS[w].capitalize()
            return r.choice([f"every {name}", f"on {name}s", f"each {name}"]), {"freq": "weekly", "by_day": [DAY_CODES[w]]}, None
        if kind == "pair":
            a, c = sorted(r.sample(range(7), 2))
            return (f"every {WEEKDAYS[a].capitalize()} and {WEEKDAYS[c].capitalize()}",
                    {"freq": "weekly", "by_day": [DAY_CODES[a], DAY_CODES[c]]}, None)
        if kind == "weekdays":
            return r.choice(["every weekday", "on weekdays", "Monday to Friday"]), {"freq": "weekly", "by_day": list(DAY_CODES[:5])}, None
        if kind == "weekend":
            return r.choice(["every weekend", "on weekends"]), {"freq": "weekly", "by_day": ["SA", "SU"]}, None
        if kind == "daily":
            return r.choice(["every day", "daily", "each day"]), {"freq": "daily"}, None
        if kind == "every_n":
            n = r.choice([2, 3, 4, 5, 7, 10])
            phrase = "every other day" if n == 2 and r.random() < 0.5 else f"every {n} days"
            return phrase, {"freq": "daily", "interval": n}, None
        n = r.choice([1, 1, 5, 10, 15, 20, 25, 28])
        phrases = [f"on the {ordinal(n)} of every month", f"every month on the {ordinal(n)}", f"monthly on the {ordinal(n)}"]
        if n == 1:
            phrases.append("on the first day of every month")
        return r.choice(phrases), {"freq": "monthly", "by_month_day": [n]}, None

    # --- scenarios --------------------------------------------------------------------

    def finish_create(self, b: Builder, res: dict[str, Any], repeat: dict[str, Any] | None, skills: list[str]) -> None:
        b.act(Action.finish(created_msg(res, repeat)), skills + ["completion_verification"],
              verify=f"scheduler.create succeeded (task_id {res['task_id']}).")

    def create_with_errors(self, b: Builder, args: dict[str, Any], plan: str, repeat, skills: list[str]) -> None:
        """Execute scheduler.create honouring injected failures (retry once)."""
        ok, res = b.act(Action.tool_call("scheduler.create", args), skills + ["tool_selection", "argument_generation", "scheduling"], plan=plan)
        if ok:
            return self.finish_create(b, res, repeat, skills)
        if res.get("retryable"):
            ok, res = b.act(Action.tool_call("scheduler.create", args), skills + ["error_recovery", "argument_generation"],
                            verify="The scheduler failed with a retryable error; retry once.")
            if ok:
                return self.finish_create(b, res, repeat, skills + ["error_recovery"])
            b.act(Action.fail("The scheduler is unavailable, so the reminder was not created. Please try again later."),
                  skills + ["error_recovery", "completion_verification"], verify="Both attempts failed; the reminder was not created.")
            return None
        if res["error"] == "that time has already passed":
            t = args["time"]
            b.act(Action.ask_user(f"{t} today has already passed. Should I set it for tomorrow at {t} instead?"),
                  skills + ["error_recovery", "ambiguity"], verify="That time has already passed today.")
            if self.rng.random() < 0.75:
                b.user(self.rng.choice(YES))
                tomorrow = (b.now + timedelta(days=1)).strftime("%Y-%m-%d")
                args2 = {**args, "date": tomorrow}
                ok, res = b.act(Action.tool_call("scheduler.create", args2), ["task_state", "argument_generation", "scheduling"],
                                plan=f"Tomorrow is {tomorrow}. One-time reminder on {tomorrow} at {t}.")
                if ok:
                    self.finish_create(b, res, None, ["task_state"])
            else:
                b.user(self.rng.choice(NO))
                b.act(Action.finish("Okay, I did not set the reminder."), ["task_state", "intent"])
            return None
        raise AssertionError(f"unexpected tool error in generator: {res}")

    def s_one_time(self) -> str:
        r = self.rng
        b = self._b
        task = self.task()
        h, m = self.time()
        t = say_time(r, h, m)
        if r.random() < (0.12 if self.test else 0.2):  # "today": may already have passed -> error path
            day, date, label = "today", b.now.strftime("%Y-%m-%d"), "Today"
        elif r.random() < 0.15:  # time only -> scheduler picks the next occurrence
            day, date, label = None, None, None
        else:
            day, date, label = self.day(b)
        if day is None:
            text = self.pick([f"Remind me at {t} to {task}.", f"Remind me to {task} at {t}.", f"At {t}, remind me to {task}.",
                              f"Set a reminder for {t} to {task}."],
                             [f"Ping me at {t} so I {task}.", f"I have to {task} at {t}, please remind me."])
            args = {"task": task, "time": hm(h, m)}
            plan = f"One-time reminder at the next {hm(h, m)}."
        else:
            text = self.pick(
                [f"Remind me {day} at {t} to {task}.", f"remind me {day} at {t} to {task}", f"{cap(day)} at {t}, remind me to {task}.",
                 f"I want a reminder {day} at {t} to {task}.", f"Set an alarm {day} at {t} to {task}.",
                 f"Make a reminder for {day} at {t} to {task}.", f"{cap(day)} at {t} remind me to {task} please.",
                 f"Remind me {day} at {t} that I have to {task}.", f"Book a reminder {day} at {t}: {task}.",
                 f"Set a reminder to {task} {day} at {t}.", f"Can you remind me to {task} {day} at {t}?",
                 f"Please remind me to {task} at {t} {day}.", f"I need to {task} {day} at {t}. Remind me.",
                 f"Create a reminder for {day} at {t}: {task}.", f"Remind me to {task} at {t} {day}.", f"Reminder {day} {t} {task}",
                 f"Could you set a reminder to {task} {day} at {t}?", f"{cap(day)} at {t} I have to {task}, remind me.",
                 f"Add a reminder {day} at {t} to {task}.", f"Schedule a reminder to {task} {day} at {t}.",
                 f"Set up a reminder {day} at {t}: {task}.", f"Remind me to {task}, {day} at {t}."],
                [f"Hey, could you set a reminder {day} at {t} so I remember to {task}?", f"Don't let me forget to {task} {day} at {t}.",
                 f"{cap(day)} {t}: {task}. Please remind me.", f"Ping me {day} at {t} to {task}."])
            args = {"task": task, "date": date, "time": hm(h, m)}
            plan = f"{label} is {date}. One-time reminder on {date} at {hm(h, m)}."
        self.say(text)
        self.create_with_errors(b, args, plan, None, ["intent"])
        return "scheduling_one_time"

    def s_relative(self) -> str:
        r, b = self.rng, self._b
        task = self.task()
        kind = r.choice(["min", "hour", "hour", "an_hour", "half"])
        if kind == "min":
            n = r.choice([5, 10, 15, 20, 25, 30, 40, 45, 50, 90])
            span, minutes = f"{n} minutes", n
        elif kind == "hour":
            n = r.randint(2, 12)
            span, minutes = f"{n} hours", 60 * n
        elif kind == "an_hour":
            span, minutes = "an hour", 60
        else:
            span, minutes = "half an hour", 30
        text = self.pick([f"Remind me in {span} to {task}.", f"In {span}, remind me to {task}.", f"Remind me to {task} in {span}.",
                          f"After {span} remind me to {task}.", f"Set a reminder in {span}: {task}.",
                          f"Can you remind me in {span} to {task}?", f"Remind me to {task} after {span}.",
                          f"In {span} I need to {task}, remind me.", f"Set a timer for {span} to {task}.",
                          f"Remind me in {span} that I have to {task}."],
                         [f"Ping me in {span} to {task}.", f"{cap(span)} from now, remind me to {task}.",
                          f"Give me a reminder to {task} in {span}."])
        self.say(text)
        self.create_with_errors(b, {"task": task, "in_minutes": minutes}, f"Relative reminder in {minutes} minutes.", None, ["intent"])
        return "scheduling_relative"

    def s_recurring(self) -> str:
        r, b = self.rng, self._b
        task = self.task()
        phrase, repeat, _ = self.rule()
        h, m = self.time()
        t = say_time(r, h, m)
        text = self.pick([f"{cap(phrase)} at {t}, remind me to {task}.", f"Remind me {phrase} at {t} to {task}.",
                          f"Remind me to {task} {phrase} at {t}.", f"Set a recurring reminder {phrase} at {t}: {task}.",
                          f"I want a reminder {phrase} at {t} to {task}.", f"Please remind me to {task} {phrase} at {t}.",
                          f"Add a reminder {phrase} at {t} to {task}.", f"Set a reminder {phrase} at {t} to {task}.",
                          f"Can you remind me to {task} {phrase} at {t}?", f"{cap(phrase)} at {t}: {task}.",
                          f"I need to {task} {phrase} at {t}, remind me."],
                         [f"Could you remind me {phrase} at {t} to {task}?", f"{cap(phrase)} at {t} I need to {task}. Remind me."])
        self.say(text)
        args = {"task": task, "time": hm(h, m), "repeat": repeat}
        self.create_with_errors(b, args, f"Recurring reminder {rule_words(repeat)} at {hm(h, m)}.", repeat, ["intent"])
        return "scheduling_recurring"

    def s_missing_time(self) -> str:
        """Ambiguous: no clock time -> ask; then (usually) the user answers and Arouse continues."""
        r, b = self.rng, self._b
        task = self.task()
        kind = r.choice(["day", "day", "rule", "vague", "later"])
        if kind == "day":
            day, date, label = self.day(b)
            text = self.pick([f"Remind me {day} to {task}.", f"Set a reminder {day} to {task}.", f"{cap(day)}, remind me to {task}.",
                              f"I need a reminder {day} to {task}.", f"Please remind me {day} to {task}.",
                              f"Can you set a reminder {day} to {task}?", f"{cap(day)} I have to {task}, remind me.",
                              f"Add a reminder {day} to {task}.", f"Remind me to {task} {day}."],
                             [f"Don't let me forget to {task} {day}.", f"Ping me {day} to {task}."])
            q, pending = Q_TIME, ("date", date, label)
        elif kind == "rule":
            phrase, repeat, _ = self.rule()
            text = self.pick([f"Remind me {phrase} to {task}.", f"{cap(phrase)}, remind me to {task}.",
                              f"Remind me to {task} {phrase}.", f"Set a reminder {phrase} to {task}.",
                              f"I want a reminder {phrase} to {task}."],
                             [f"Could you remind me {phrase} to {task}?"])
            q, pending = Q_TIME, ("repeat", repeat, None)
        elif kind == "vague":
            day, date, label = self.day(b)
            vague = r.choice(["in the morning", "in the evening", "after dinner", "in the afternoon", "after lunch"])
            text = self.pick([f"Remind me {day} {vague} to {task}.", f"{cap(day)} {vague}, remind me to {task}.",
                              f"Remind me to {task} {day} {vague}.", f"Set a reminder {day} {vague} to {task}."],
                             [f"Ping me {day} {vague} to {task}."])
            q, pending = Q_TIME, ("date", date, label)
        else:
            text = self.pick([f"Remind me later to {task}.", f"Remind me to {task}.", f"Set a reminder to {task}.",
                              f"I need a reminder to {task}.", f"Add a reminder to {task}.", f"Please remind me to {task}."],
                             [f"Can you remind me to {task} sometime?"])
            q, pending = Q_WHEN, None
        self.say(text)
        b.act(Action.ask_user(q), ["intent", "ambiguity"], plan="The time is missing, so I should ask instead of guessing.")
        if pending is None or r.random() < 0.3:
            return "ambiguity"
        h, m = self.time()
        t = say_time(r, h, m)
        b.user(r.choice([t, f"at {t}", f"make it {t}", f"{t} please"]))
        if pending[0] == "date":
            args = {"task": task, "date": pending[1], "time": hm(h, m)}
            plan = f"{pending[2]} is {pending[1]}. One-time reminder on {pending[1]} at {hm(h, m)}."
            rep = None
        else:
            args = {"task": task, "time": hm(h, m), "repeat": pending[1]}
            plan = f"Recurring reminder {rule_words(pending[1])} at {hm(h, m)}."
            rep = pending[1]
        self.create_with_errors(b, args, plan, rep, ["task_state"])
        return "ambiguity_followup"

    def s_missing_task(self) -> str:
        r, b = self.rng, self._b
        h, m = self.time()
        t = say_time(r, h, m)
        day, date, label = self.day(b)
        self.say(self.pick([f"Set a reminder for {day} at {t}.", f"Remind me {day} at {t}.", f"Create a reminder {day} at {t}.",
                          f"Add a reminder {day} at {t}.", f"Remind me of something {day} at {t}."],
                         [f"I need a reminder {day} at {t}."]))
        b.act(Action.ask_user(Q_TASK), ["intent", "ambiguity"], plan="The reminder text is missing, so I should ask.")
        if r.random() < 0.3:
            return "ambiguity"
        task = self.task()
        reply, task = r.choice([(task, task), (f"to {task}", task), (f"It's to {task}", task), (cap(task), cap(task))])
        b.user(reply)
        self.create_with_errors(b, {"task": task, "date": date, "time": hm(h, m)},
                                f"{label} is {date}. One-time reminder on {date} at {hm(h, m)}.", None, ["task_state"])
        return "ambiguity_followup"

    def s_list(self) -> str:
        b = self._b
        self.say(self.pick(["What reminders do I have?", "Show my reminders.", "List my reminders", "What's on my schedule?",
                          "Do I have any reminders?", "show reminders", "What have I scheduled?", "my reminders",
                          "What are my reminders?", "Can you list my reminders?", "Show me everything I scheduled."],
                         ["Which reminders are set?", "Tell me my upcoming reminders."]))
        ok, res = b.act(Action.tool_call("scheduler.list", {}), ["intent", "tool_selection"], plan="List the reminders.")
        items = res["reminders"]
        if not items:
            msg = "You have no reminders."
        else:
            parts = [f"{x['task']} on {x['next_run'].replace('T', ' at ')}" + (" (repeats)" if "repeat" in x else "") for x in items]
            msg = f"You have {len(items)} reminder{'s' if len(items) > 1 else ''}: " + "; ".join(parts) + "."
        b.act(Action.finish(msg), ["multi_step", "completion_verification"], verify=f"scheduler.list returned {len(items)}.")
        return "list"

    def s_delete(self) -> str:
        r, b = self.rng, self._b
        existing = list(b.sandbox.reminders.values())
        if existing and r.random() < 0.8:
            target = r.choice(existing).task
        else:
            taken = {x.task.lower() for x in existing}
            target = next(t for t in iter(self.task, None) if t.lower() not in taken)
        q = target
        self.say(self.pick([f"Cancel my reminder to {q}.", f"Delete the reminder to {q}.", f"Remove the reminder about {q}.",
                          f"Stop reminding me to {q}.", f"Delete my reminder to {q}.", f"Remove my {q} reminder.",
                          f"Cancel the {q} reminder.", f"Get rid of the reminder to {q}.",
                          f"I don't need the reminder to {q} anymore.", f"Erase the reminder about {q}.",
                          f"Turn off the reminder to {q}.", f"Can you delete the reminder about {q}?",
                          f"Clear the reminder to {q}.", f"Forget the reminder to {q}.", f"Remove {q} from my reminders.",
                          f"Delete {q} from my reminders.", f"No need to remind me to {q} anymore.",
                          f"Cancel the reminder about {q}, please."],
                         [f"I no longer need the reminder to {q}. Delete it.", f"Please drop the {q} reminder."]))
        ok, res = b.act(Action.tool_call("scheduler.list", {}), ["intent", "tool_selection"], plan="Find the reminder, then delete it.")
        match = [x for x in res["reminders"] if x["task"].lower() == target.lower()]
        if not match:
            b.act(Action.fail(f"I couldn't find a reminder to {q}."), ["multi_step", "completion_verification"],
                  verify=f"No reminder matches '{target}'.")
            return "delete"
        tid = match[0]["task_id"]
        ok, res = b.act(Action.tool_call("scheduler.delete", {"task_id": tid}), ["multi_step", "argument_generation", "task_state"],
                        verify=f"'{target}' has task_id {tid}.")
        b.act(Action.finish(f"Deleted the reminder: {res['task']}."), ["completion_verification"],
              verify=f"scheduler.delete succeeded for {tid}.")
        return "delete"

    def s_note(self) -> str:
        b = self._b
        fact = self.fact()
        self.say(self.pick([f"Note that {fact}.", f"Save a note: {fact}.", f"Write down that {fact}.", f"Make a note: {fact}.",
                          f"Add a note: {fact}.", f"Please note: {fact}.", f"Keep a note that {fact}.", f"Record this: {fact}.",
                          f"Take a note: {fact}.", f"note: {fact}", f"Write this down: {fact}.", f"Can you note that {fact}?",
                          f"Add to my notes: {fact}.", f"Put in my notes that {fact}.", f"Save this: {fact}.",
                          f"Log this: {fact}.", f"Note down that {fact}.", f"Make a note that {fact}.",
                          f"Please save a note: {fact}.", f"Store a note: {fact}.", f"New note: {fact}.",
                          f"I want to note that {fact}.", f"Could you write down that {fact}?", f"Memo: {fact}.",
                          f"Create a note saying {fact}.", f"Add a note saying {fact}.", f"Note this down: {fact}.",
                          f"For my notes: {fact}.", f"Write a note: {fact}.", f"Keep this in my notes: {fact}."],
                         [f"Please jot down: {fact}.", f"Remember this note: {fact}."]))
        ok, res = b.act(Action.tool_call("notes.create", {"text": fact}), ["intent", "tool_selection", "argument_generation"],
                        plan="Save a note.")
        if ok:
            b.act(Action.finish("Saved the note."), ["completion_verification"], verify=f"notes.create succeeded ({res['note_id']}).")
            return "note"
        ok, res = b.act(Action.tool_call("notes.create", {"text": fact}), ["error_recovery"],
                        verify="Saving failed with a retryable error; retry once.")
        if ok:
            b.act(Action.finish("Saved the note."), ["completion_verification", "error_recovery"], verify=f"notes.create succeeded ({res['note_id']}).")
        else:
            b.act(Action.fail("I couldn't save the note because the notes service is unavailable."),
                  ["error_recovery", "completion_verification"], verify="Both attempts failed; the note was not saved.")
        return "note"

    def s_file(self) -> str:
        r, b = self.rng, self._b
        files = sorted(b.sandbox.files)
        missing = r.random() < 0.35
        if missing:
            base = r.choice(files)
            stem, ext = base.rsplit(".", 1)
            similar = r.random() < 0.6 and "_" in stem
            name = stem.split("_")[0] + "." + ext if similar else next(f for f in iter(lambda: self.file_names(1)[0], None) if f not in files)
            if name in files:
                missing, name = False, name
        else:
            name = r.choice(files)
        count = r.random() < 0.4
        text = self.pick([f"How many lines are in {name}?", f"How many rows does {name} have?",
                          f"How many lines does {name} have?", f"How long is {name}?", f"Number of lines in {name}?",
                          f"Tell me how many rows {name} has.", f"How many entries are in {name}?"],
                         [f"Count the lines in {name}."]) if count else \
            self.pick([f"Read {name}.", f"Open {name}.", f"What's in {name}?", f"Show me {name}.", f"Read {name} for me.",
                       f"Open the file {name}.", f"Show {name}.", f"What does {name} say?", f"Display {name}."],
                      [f"Can you look at {name}?"])
        self.say(text)
        ok, res = b.act(Action.tool_call("file.read", {"path": name}), ["intent", "tool_selection", "argument_generation"],
                        plan=f"Read {name}.")

        def done(res: dict[str, Any], skills: list[str]) -> None:
            first = res["preview"].split("\n")[0]
            msg = f"{res['path']} has {res['lines']} lines." + ("" if count else f" It starts with: {first}")
            b.act(Action.finish(msg), skills + ["completion_verification"], verify=f"file.read succeeded for {res['path']}.")

        if ok:
            done(res, [])
            return "file"
        ok, lst = b.act(Action.tool_call("file.list", {}), ["error_recovery", "multi_step", "tool_selection"],
                        verify="The file does not exist; list the files.")
        stem = name.rsplit(".", 1)[0]
        cands = [f for f in lst["files"] if f.startswith(stem + "_")]
        if cands:
            alt = cands[0]
            b.act(Action.ask_user(f"I couldn't find {name}. Did you mean {alt}?"), ["error_recovery", "ambiguity"],
                  verify=f"{alt} looks like the right file.")
            if r.random() < 0.7:
                b.user(r.choice(YES))
                ok, res = b.act(Action.tool_call("file.read", {"path": alt}), ["task_state", "argument_generation"], plan=f"Read {alt}.")
                done(res, ["task_state"])
            return "file_error"
        b.act(Action.ask_user(f"I couldn't find {name}. Which file should I use? Available: {', '.join(lst['files'])}."),
              ["error_recovery", "ambiguity"], verify="No similar file exists.")
        return "file_error"

    def s_list_files(self) -> str:
        b = self._b
        self.say(self.pick(["What files do I have?", "List my files.", "Show my files", "Which files are there?",
                          "What files are available?", "show files", "List all files."],
                         ["What files can you see?"]))
        ok, res = b.act(Action.tool_call("file.list", {}), ["intent", "tool_selection"], plan="List the files.")
        b.act(Action.finish(f"You have {len(res['files'])} files: {', '.join(res['files'])}."), ["completion_verification"],
              verify="file.list succeeded.")
        return "file"

    def s_chat(self) -> str:
        r = self.rng
        if self.test:  # exact original draw order: the held-out benchmark must not change
            kind = r.choices(["greet", "help", "who", "oos"], [3, 2, 1, 4])[0]
            text, reply = {
                "greet": (r.choice(self.greetings), R_GREET),
                "help": (r.choice(self.help), R_HELP),
                "who": (r.choice(WHO), R_WHO),
                "oos": (r.choice(self.oos), R_OUT),
            }[kind]
        else:
            kind = r.choices(["greet", "help", "who", "oos", "how", "bye", "ack"], [3, 2, 1, 8, 1, 1, 1])[0]
            if kind == "oos":
                banned = {t.lower() for t in TEST_OUT_OF_SCOPE} | {"tell me a story"}
                text = r.choice(self.oos) if r.random() < 0.25 else next(
                    q for q in iter(lambda: oos_question(r), None)
                    if q.lower() not in banned and "mumbai" not in q.lower() and "tata" not in q.lower())
                reply = R_OUT
            else:
                pool, reply = {"greet": (self.greetings, R_GREET), "help": (self.help, R_HELP), "who": (WHO, R_WHO),
                               "how": (HOW_ARE_YOU, R_HOW), "bye": (BYE, R_BYE), "ack": (ACKS, R_ACK)}[kind]
                text = r.choice(pool)
        if r.random() < 0.3:
            text = cap(text)
        self.say(text, chat=kind != "oos")
        self._b.act(Action.finish(reply), ["intent"])
        return "chat" if kind != "oos" else "out_of_scope"

    TRAIN_SCENARIOS: list[tuple[str, int]] = [
        ("s_one_time", 18), ("s_relative", 5), ("s_recurring", 10), ("s_missing_time", 14), ("s_missing_task", 4),
        ("s_list", 5), ("s_delete", 9), ("s_note", 8), ("s_file", 10), ("s_list_files", 2), ("s_chat", 12),
    ]

    SCENARIOS: list[tuple[str, int]] = [
        ("s_one_time", 16), ("s_relative", 6), ("s_recurring", 12), ("s_missing_time", 12), ("s_missing_task", 4),
        ("s_list", 6), ("s_delete", 6), ("s_note", 5), ("s_file", 8), ("s_list_files", 2), ("s_chat", 8),
    ]

    def episode(self, eid: str) -> dict[str, Any]:
        r = self.rng
        failures = {}
        if r.random() < 0.12:
            failures["scheduler.create"] = r.choice([1, 1, 2])
        if r.random() < 0.03:
            failures["notes.create"] = r.choice([1, 2])
        self._b = b = self.world(failures)
        names, weights = zip(*(self.SCENARIOS if self.test else self.TRAIN_SCENARIOS))
        cats = []
        for i in range(r.choices([1, 2, 3], [70, 22, 8])[0]):
            cats.append(getattr(self, r.choices(names, weights)[0])())
            if i == 0 and r.random() < 0.1:
                b.user(r.choice(THANKS))
                b.act(Action.finish(R_THANKS), ["intent"])
        return b.to_dict(eid, "+".join(cats), self.split)


def generate(n: int, seed: int, split: str = "train", on_episode: Callable[[dict], None] | None = None) -> list[dict[str, Any]]:
    g = Generator(seed, split)
    out = []
    for i in range(n):
        ep = g.episode(f"{split}-{seed}-{i}")
        if on_episode:
            on_episode(ep)
        out.append(ep)
    return out
