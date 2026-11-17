"""
generate_dataset.py
-------------------
Generates 700+ natural-language → JSON training examples for a calendar
assistant, covering 6 action types. Saves full dataset + train/val/test splits.
"""
import json
import random
import os
from datetime import datetime, timedelta

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
    "Ivy", "Jack", "Karen", "Leo", "Mia", "Nina", "Oscar", "Pam",
    "Quinn", "Ray", "Sara", "Tom", "Uma", "Vince", "Wendy", "Xander",
    "Yara", "Zoe", "Liam", "Noah", "Emma", "Olivia", "Ava", "Sophia",
    "James", "Lucas", "Mason", "Ethan", "Aiden", "Chloe", "Isabella",
]

LOCATIONS = [
    "Conference Room A", "Conference Room B", "Main Hall", "Zoom", "Teams",
    "Google Meet", "the office", "cafeteria", "rooftop terrace", "lobby",
    "room 202", "auditorium", "board room", "my place", "their office",
    "downtown HQ", "satellite campus", "the lab", "Studio 3", "Park",
]

TASK_NAMES = [
    "submit the report", "review pull requests", "prepare slides",
    "send the invoice", "update the roadmap", "call the client",
    "finish the analysis", "write the blog post", "deploy the fix",
    "read the proposal", "complete onboarding", "schedule interviews",
    "audit the logs", "update dependencies", "run the tests",
    "draft the email", "file the expenses", "book the flight",
    "renew the license", "back up the database",
]

PRIORITIES = ["low", "medium", "high", "urgent"]

MESSAGES = [
    "I'll be 10 minutes late", "can we reschedule?", "please confirm receipt",
    "the meeting link is in your calendar", "bring the signed docs",
    "looking forward to our call", "here are the notes from last time",
    "the deadline has changed to next Friday", "please review before EOD",
    "just following up on this", "great job on the presentation",
    "can you join 5 minutes early?", "sending the agenda now",
    "the venue has changed", "please prepare a 2-minute intro",
]


def random_date(start_offset=1, end_offset=60):
    """Return a random future date string in YYYY-MM-DD format."""
    base = datetime.today() + timedelta(days=random.randint(start_offset, end_offset))
    return base.strftime("%Y-%m-%d")


def random_time():
    """Return a random time string in HH:MM format (business hours)."""
    hour = random.randint(8, 18)
    minute = random.choice([0, 15, 30, 45])
    return f"{hour:02d}:{minute:02d}"


def r_name():
    return random.choice(NAMES)


def r_loc():
    return random.choice(LOCATIONS)


def r_task():
    return random.choice(TASK_NAMES)


def r_priority():
    return random.choice(PRIORITIES)


def r_msg():
    return random.choice(MESSAGES)


# ── 1. create_event ───────────────────────────────────────────────────────────

CREATE_EVENT_TEMPLATES = [
    lambda n, d, t, l: (
        f"Schedule a meeting with {n} on {d} at {t} in {l}.",
        {"action": "create_event", "person": n, "date": d, "time": t, "location": l},
    ),
    lambda n, d, t, l: (
        f"Set up a call with {n} for {d} at {t}.",
        {"action": "create_event", "person": n, "date": d, "time": t},
    ),
    lambda n, d, t, l: (
        f"Create an appointment with {n} on {d} at {t} at {l}.",
        {"action": "create_event", "person": n, "date": d, "time": t, "location": l},
    ),
    lambda n, d, t, l: (
        f"Book a session with {n} at {t} on {d} in {l}.",
        {"action": "create_event", "person": n, "date": d, "time": t, "location": l},
    ),
    lambda n, d, t, l: (
        f"I need to meet {n} on {d} at {t}.",
        {"action": "create_event", "person": n, "date": d, "time": t},
    ),
    lambda n, d, t, l: (
        f"Can you add an event: meet {n} at {l} on {d} at {t}?",
        {"action": "create_event", "person": n, "date": d, "time": t, "location": l},
    ),
    lambda n, d, t, l: (
        f"Plan a meeting with {n} in {l} at {t} on {d}.",
        {"action": "create_event", "person": n, "date": d, "time": t, "location": l},
    ),
    lambda n, d, t, l: (
        f"Add {n} to my calendar on {d} at {t}.",
        {"action": "create_event", "person": n, "date": d, "time": t},
    ),
    lambda n, d, t, l: (
        f"Organise a {t} meeting with {n} on {d} in {l}.",
        {"action": "create_event", "person": n, "date": d, "time": t, "location": l},
    ),
    lambda n, d, t, l: (
        f"I want to catch up with {n} on {d} at {t} at {l}.",
        {"action": "create_event", "person": n, "date": d, "time": t, "location": l},
    ),
    lambda n, d, t, l: (
        f"Put a meeting with {n} in my calendar for {d} at {t}.",
        {"action": "create_event", "person": n, "date": d, "time": t},
    ),
    lambda n, d, t, l: (
        f"Let's sync with {n} on {d} at {t} at {l}.",
        {"action": "create_event", "person": n, "date": d, "time": t, "location": l},
    ),
    lambda n, d, t, l: (
        f"Schedule a {t} stand-up with {n} on {d} in {l}.",
        {"action": "create_event", "person": n, "date": d, "time": t, "location": l},
    ),
    lambda n, d, t, l: (
        f"New event: {n}, {d}, {t}.",
        {"action": "create_event", "person": n, "date": d, "time": t},
    ),
    lambda n, d, t, l: (
        f"I'd like to schedule time with {n} at {l} on {d} starting at {t}.",
        {"action": "create_event", "person": n, "date": d, "time": t, "location": l},
    ),
]


def gen_create_event(n=130):
    examples = []
    for _ in range(n):
        name = r_name()
        date = random_date()
        time_ = random_time()
        loc = r_loc()
        tmpl = random.choice(CREATE_EVENT_TEMPLATES)
        inp, out = tmpl(name, date, time_, loc)
        examples.append({"input": inp, "output": out})
    return examples


# ── 2. set_reminder ───────────────────────────────────────────────────────────

SET_REMINDER_TEMPLATES = [
    lambda t, d, tm: (
        f"Remind me to {t} on {d} at {tm}.",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
    lambda t, d, tm: (
        f"Set a reminder for {t} at {tm} on {d}.",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
    lambda t, d, tm: (
        f"Don't let me forget to {t} on {d} at {tm}.",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
    lambda t, d, tm: (
        f"Add a reminder: {t} — {d} at {tm}.",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
    lambda t, d, tm: (
        f"Can you ping me to {t} on {d} at {tm}?",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
    lambda t, d, tm: (
        f"I need a reminder to {t} at {tm} on {d}.",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
    lambda t, d, tm: (
        f"Please remind me about {t} on {d} at {tm}.",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
    lambda t, d, tm: (
        f"Set an alert for {d} at {tm} to {t}.",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
    lambda t, d, tm: (
        f"Schedule a notification at {tm} on {d} to {t}.",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
    lambda t, d, tm: (
        f"Notify me to {t} when it is {tm} on {d}.",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
    lambda t, d, tm: (
        f"Create a reminder for me: {t} on {d} at {tm}.",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
    lambda t, d, tm: (
        f"I want to be reminded to {t} at {tm} on {d}.",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
    lambda t, d, tm: (
        f"Flag {d} at {tm} so I don't miss {t}.",
        {"action": "set_reminder", "task": t, "date": d, "time": tm},
    ),
]


def gen_set_reminder(n=120):
    examples = []
    for _ in range(n):
        task = r_task()
        date = random_date()
        time_ = random_time()
        tmpl = random.choice(SET_REMINDER_TEMPLATES)
        inp, out = tmpl(task, date, time_)
        examples.append({"input": inp, "output": out})
    return examples


# ── 3. cancel_event ───────────────────────────────────────────────────────────

CANCEL_EVENT_TEMPLATES = [
    lambda n, d: (
        f"Cancel my meeting with {n} on {d}.",
        {"action": "cancel_event", "person": n, "date": d},
    ),
    lambda n, d: (
        f"Remove the event with {n} from {d}.",
        {"action": "cancel_event", "person": n, "date": d},
    ),
    lambda n, d: (
        f"I can't make it to the meeting with {n} on {d}. Cancel it.",
        {"action": "cancel_event", "person": n, "date": d},
    ),
    lambda n, d: (
        f"Delete the appointment with {n} scheduled for {d}.",
        {"action": "cancel_event", "person": n, "date": d},
    ),
    lambda n, d: (
        f"Call off the {d} session with {n}.",
        {"action": "cancel_event", "person": n, "date": d},
    ),
    lambda n, d: (
        f"Please cancel my {d} meeting with {n}.",
        {"action": "cancel_event", "person": n, "date": d},
    ),
    lambda n, d: (
        f"Scratch the meeting with {n} on {d}.",
        {"action": "cancel_event", "person": n, "date": d},
    ),
    lambda n, d: (
        f"I need to cancel the event with {n} on {d}.",
        {"action": "cancel_event", "person": n, "date": d},
    ),
    lambda n, d: (
        f"Drop the {d} meeting with {n}.",
        {"action": "cancel_event", "person": n, "date": d},
    ),
    lambda n, d: (
        f"Unschedule my appointment with {n} on {d}.",
        {"action": "cancel_event", "person": n, "date": d},
    ),
    lambda n, d: (
        f"The meeting with {n} on {d} is no longer happening. Remove it.",
        {"action": "cancel_event", "person": n, "date": d},
    ),
    lambda n, d: (
        f"Free up {d} — cancel my time with {n}.",
        {"action": "cancel_event", "person": n, "date": d},
    ),
]


def gen_cancel_event(n=110):
    examples = []
    for _ in range(n):
        name = r_name()
        date = random_date()
        tmpl = random.choice(CANCEL_EVENT_TEMPLATES)
        inp, out = tmpl(name, date)
        examples.append({"input": inp, "output": out})
    return examples


# ── 4. reschedule_event ───────────────────────────────────────────────────────

RESCHEDULE_TEMPLATES = [
    lambda n, od, nd, nt: (
        f"Move my meeting with {n} from {od} to {nd} at {nt}.",
        {"action": "reschedule_event", "person": n, "old_date": od, "new_date": nd, "new_time": nt},
    ),
    lambda n, od, nd, nt: (
        f"Reschedule the appointment with {n} originally on {od} to {nd} at {nt}.",
        {"action": "reschedule_event", "person": n, "old_date": od, "new_date": nd, "new_time": nt},
    ),
    lambda n, od, nd, nt: (
        f"Change my {od} meeting with {n} to {nd} at {nt}.",
        {"action": "reschedule_event", "person": n, "old_date": od, "new_date": nd, "new_time": nt},
    ),
    lambda n, od, nd, nt: (
        f"Shift the {od} event with {n} to {nd} at {nt}.",
        {"action": "reschedule_event", "person": n, "old_date": od, "new_date": nd, "new_time": nt},
    ),
    lambda n, od, nd, nt: (
        f"I need to push my {n} meeting from {od} to {nd} at {nt}.",
        {"action": "reschedule_event", "person": n, "old_date": od, "new_date": nd, "new_time": nt},
    ),
    lambda n, od, nd, nt: (
        f"Please update the {n} meeting: was {od}, now {nd} at {nt}.",
        {"action": "reschedule_event", "person": n, "old_date": od, "new_date": nd, "new_time": nt},
    ),
    lambda n, od, nd, nt: (
        f"Can we bump the {od} call with {n} to {nd} at {nt}?",
        {"action": "reschedule_event", "person": n, "old_date": od, "new_date": nd, "new_time": nt},
    ),
    lambda n, od, nd, nt: (
        f"Postpone my {od} meeting with {n} to {nd} at {nt}.",
        {"action": "reschedule_event", "person": n, "old_date": od, "new_date": nd, "new_time": nt},
    ),
    lambda n, od, nd, nt: (
        f"Move the {n} session on {od} to a new time: {nd} at {nt}.",
        {"action": "reschedule_event", "person": n, "old_date": od, "new_date": nd, "new_time": nt},
    ),
    lambda n, od, nd, nt: (
        f"Update my meeting with {n}: old date {od}, new date {nd} at {nt}.",
        {"action": "reschedule_event", "person": n, "old_date": od, "new_date": nd, "new_time": nt},
    ),
    lambda n, od, nd, nt: (
        f"The {od} meeting with {n} needs to move to {nd} at {nt}.",
        {"action": "reschedule_event", "person": n, "old_date": od, "new_date": nd, "new_time": nt},
    ),
    lambda n, od, nd, nt: (
        f"Reschedule {n}'s meeting: {od} → {nd} at {nt}.",
        {"action": "reschedule_event", "person": n, "old_date": od, "new_date": nd, "new_time": nt},
    ),
]


def gen_reschedule_event(n=120):
    examples = []
    for _ in range(n):
        name = r_name()
        old_date = random_date(1, 30)
        new_date = random_date(31, 60)
        new_time = random_time()
        tmpl = random.choice(RESCHEDULE_TEMPLATES)
        inp, out = tmpl(name, old_date, new_date, new_time)
        examples.append({"input": inp, "output": out})
    return examples


# ── 5. create_task ────────────────────────────────────────────────────────────

CREATE_TASK_TEMPLATES = [
    lambda title, dd, pri: (
        f"Add a task: {title} due {dd} with {pri} priority.",
        {"action": "create_task", "title": title, "due_date": dd, "priority": pri},
    ),
    lambda title, dd, pri: (
        f"Create a {pri} priority task to {title} by {dd}.",
        {"action": "create_task", "title": title, "due_date": dd, "priority": pri},
    ),
    lambda title, dd, pri: (
        f"I need to {title} before {dd}. Set priority to {pri}.",
        {"action": "create_task", "title": title, "due_date": dd, "priority": pri},
    ),
    lambda title, dd, pri: (
        f"New task: {title}. Due date: {dd}. Priority: {pri}.",
        {"action": "create_task", "title": title, "due_date": dd, "priority": pri},
    ),
    lambda title, dd, pri: (
        f"Schedule task '{title}' with deadline {dd} and {pri} urgency.",
        {"action": "create_task", "title": title, "due_date": dd, "priority": pri},
    ),
    lambda title, dd, pri: (
        f"Log a {pri} task: {title} — must be done by {dd}.",
        {"action": "create_task", "title": title, "due_date": dd, "priority": pri},
    ),
    lambda title, dd, pri: (
        f"Add to my to-do: {title}. Deadline {dd}, {pri} priority.",
        {"action": "create_task", "title": title, "due_date": dd, "priority": pri},
    ),
    lambda title, dd, pri: (
        f"Remind me I have to {title} by {dd}. This is {pri}.",
        {"action": "create_task", "title": title, "due_date": dd, "priority": pri},
    ),
    lambda title, dd, pri: (
        f"Task: {title}. Due {dd}. Mark as {pri}.",
        {"action": "create_task", "title": title, "due_date": dd, "priority": pri},
    ),
    lambda title, dd, pri: (
        f"Create a task called '{title}' that is due on {dd} and has {pri} priority.",
        {"action": "create_task", "title": title, "due_date": dd, "priority": pri},
    ),
    lambda title, dd, pri: (
        f"Please add '{title}' to my task list. Due: {dd}. Priority: {pri}.",
        {"action": "create_task", "title": title, "due_date": dd, "priority": pri},
    ),
    lambda title, dd, pri: (
        f"Track a {pri} item: {title}, deadline {dd}.",
        {"action": "create_task", "title": title, "due_date": dd, "priority": pri},
    ),
]


def gen_create_task(n=120):
    examples = []
    for _ in range(n):
        title = r_task()
        due_date = random_date()
        priority = r_priority()
        tmpl = random.choice(CREATE_TASK_TEMPLATES)
        inp, out = tmpl(title, due_date, priority)
        examples.append({"input": inp, "output": out})
    return examples


# ── 6. send_message ───────────────────────────────────────────────────────────

SEND_MESSAGE_TEMPLATES = [
    lambda n, m: (
        f"Send a message to {n}: '{m}'",
        {"action": "send_message", "person": n, "message": m},
    ),
    lambda n, m: (
        f"Message {n} and say: {m}",
        {"action": "send_message", "person": n, "message": m},
    ),
    lambda n, m: (
        f"Text {n}: {m}",
        {"action": "send_message", "person": n, "message": m},
    ),
    lambda n, m: (
        f"Tell {n} that {m}",
        {"action": "send_message", "person": n, "message": m},
    ),
    lambda n, m: (
        f"Send {n} the following message: {m}",
        {"action": "send_message", "person": n, "message": m},
    ),
    lambda n, m: (
        f"Let {n} know: {m}",
        {"action": "send_message", "person": n, "message": m},
    ),
    lambda n, m: (
        f"Drop {n} a message saying {m}",
        {"action": "send_message", "person": n, "message": m},
    ),
    lambda n, m: (
        f"Ping {n} with: {m}",
        {"action": "send_message", "person": n, "message": m},
    ),
    lambda n, m: (
        f"Notify {n}: {m}",
        {"action": "send_message", "person": n, "message": m},
    ),
    lambda n, m: (
        f"Forward a note to {n}: {m}",
        {"action": "send_message", "person": n, "message": m},
    ),
    lambda n, m: (
        f"Write to {n}: {m}",
        {"action": "send_message", "person": n, "message": m},
    ),
    lambda n, m: (
        f"Chat {n} and tell them: {m}",
        {"action": "send_message", "person": n, "message": m},
    ),
    lambda n, m: (
        f"Compose a message to {n} saying '{m}'",
        {"action": "send_message", "person": n, "message": m},
    ),
]


def gen_send_message(n=120):
    examples = []
    for _ in range(n):
        name = r_name()
        msg = r_msg()
        tmpl = random.choice(SEND_MESSAGE_TEMPLATES)
        inp, out = tmpl(name, msg)
        examples.append({"input": inp, "output": out})
    return examples


# ── assemble + split ──────────────────────────────────────────────────────────

def build_dataset():
    all_examples = (
        gen_create_event(130)
        + gen_set_reminder(120)
        + gen_cancel_event(110)
        + gen_reschedule_event(120)
        + gen_create_task(120)
        + gen_send_message(120)
    )

    random.shuffle(all_examples)
    total = len(all_examples)

    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)
    # rest goes to test; ensure ≥ 60
    train = all_examples[:train_end]
    val = all_examples[train_end:val_end]
    test = all_examples[val_end:]

    # Guarantee ≥ 60 test examples
    while len(test) < 60:
        test.append(val.pop())

    print(f"Total: {total} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    assert len(test) >= 60, "Test set must have ≥ 60 examples"
    assert len(train) + len(val) + len(test) == total

    # Count by action
    def count_actions(examples):
        counts = {}
        for ex in examples:
            action = ex["output"]["action"]
            counts[action] = counts.get(action, 0) + 1
        return counts

    stats = {
        "total": total,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "action_counts": count_actions(all_examples),
        "train_action_counts": count_actions(train),
        "val_action_counts": count_actions(val),
        "test_action_counts": count_actions(test),
    }

    # Save files
    for name, data in [
        ("nl_to_json_dataset.json", all_examples),
        ("train.json", train),
        ("val.json", val),
        ("test.json", test),
    ]:
        path = os.path.join(DATASET_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {path}")

    stats_path = os.path.join(DATASET_DIR, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")

    return stats


if __name__ == "__main__":
    stats = build_dataset()
    print("\n=== Dataset Statistics ===")
    print(json.dumps(stats, indent=2))
