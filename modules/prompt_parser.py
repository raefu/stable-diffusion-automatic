import re
from collections import namedtuple
import torch

import modules.shared as shared

re_prompt = re.compile(r'''
(.*?)
\[
    ([^]:]+):
    (?:([^]:]*):)?
    ([0-9]*\.?[0-9]+)
]
|
(.+)
''', re.X)

# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][ in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25, 'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50, 'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60, 'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75, 'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']


def get_learned_conditioning_prompt_schedules(prompts, steps):
    res = []
    cache = {}

    for prompt in prompts:
        prompt_schedule: list[list[str | int]] = [[steps, ""]]

        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        for m in re_prompt.finditer(prompt):
            plaintext = m.group(1) if m.group(5) is None else m.group(5)
            concept_from = m.group(2)
            concept_to = m.group(3)
            if concept_to is None:
                concept_to = concept_from
                concept_from = ""
            swap_position = float(m.group(4)) if m.group(4) is not None else None

            if swap_position is not None:
                if swap_position < 1:
                    swap_position = swap_position * steps
                swap_position = int(min(swap_position, steps))

            swap_index = None
            found_exact_index = False
            for i in range(len(prompt_schedule)):
                end_step = prompt_schedule[i][0]
                prompt_schedule[i][1] += plaintext

                if swap_position is not None and swap_index is None:
                    if swap_position == end_step:
                        swap_index = i
                        found_exact_index = True

                    if swap_position < end_step:
                        swap_index = i

            if swap_index is not None:
                if not found_exact_index:
                    prompt_schedule.insert(swap_index, [swap_position, prompt_schedule[swap_index][1]])

                for i in range(len(prompt_schedule)):
                    end_step = prompt_schedule[i][0]
                    must_replace = swap_position < end_step

                    prompt_schedule[i][1] += concept_to if must_replace else concept_from

        res.append(prompt_schedule)
        cache[prompt] = prompt_schedule
        #for t in prompt_schedule:
        #    print(t)

    return res

weighted_prompt_parser = re.compile(
    """
    (?P<prompt>     # capture group for 'prompt'
    (?:\\\:|[^:])+  # match one or more non ':' characters or escaped colons '\:'
    )               # end 'prompt'
    (?:             # non-capture group
    :+              # match one or more ':' characters
    (?P<weight>     # capture group for 'weight'
    -?\d+(?:\.\d+)? # match positive or negative integer or decimal number
    )?              # end weight capture group, make optional
    \s*             # strip spaces after weight
    |               # OR
    $               # else, if no ':' then match end of line
    )               # end non-capture group
""", re.VERBOSE)

# grabs all text up to the first occurrence of ':' as sub-prompt
# takes the value following ':' as weight
# if ':' has no value defined, defaults to 1.0
# repeats until no text remaining
def split_weighted_subprompts(input_string, normalize=True):
    parsed_prompts = [(match.group("prompt").replace("\\:", ":"),
                       float(match.group("weight") or 1))
                      for match in re.finditer(weighted_prompt_parser, input_string)]
    if not input_string:
        return []
    if any(x[1] <= 0 for x in parsed_prompts):
        # normalization with negative weights doesn't make sense
        normalize = False
    if not normalize:
        return parsed_prompts
    weight_sum = sum(x[1] for x in parsed_prompts)
    if weight_sum == 0:
        print(
            "Warning: Subprompt weights add up to zero. Discarding and using even weights instead."
        )
        equal_weight = 1 / (len(parsed_prompts) or 1)
        return [(x[0], equal_weight) for x in parsed_prompts]
    if len(parsed_prompts) > 1:
        print('\n'.join(f'{weight / weight_sum} x {prompt}' for prompt, weight in parsed_prompts))
    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]

ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])
ScheduledPromptBatch = namedtuple("ScheduledPromptBatch", ["shape", "schedules"])


def get_learned_conditioning(model, prompts, steps):

    res = []

    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps)
    cache = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):

        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        texts = [x[1] for x in prompt_schedule]
        conds = []

        for text in texts:
            weighted_subprompts = split_weighted_subprompts(text)
            if len(weighted_subprompts) <= 1:
                c = model.get_learned_conditioning([text])
            else:
                c = None
                for subtext, subweight in weighted_subprompts:
                    if c is None:
                        c = model.get_learned_conditioning([subtext])
                        c *= subweight
                    else:
                        c.add_(model.get_learned_conditioning([subtext]), alpha=subweight)
            conds.append(c[0])

        cond_schedule = []
        for i, (end_at_step, text) in enumerate(prompt_schedule):
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, conds[i]))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)

    return ScheduledPromptBatch((len(prompts),) + res[0][0].cond.shape, res)


def reconstruct_cond_batch(c: ScheduledPromptBatch, current_step):
    res = torch.zeros(c.shape, device=shared.device, dtype=next(shared.sd_model.parameters()).dtype)
    for i, cond_schedule in enumerate(c.schedules):
        target_index = 0
        for curret_index, (end_at, cond) in enumerate(cond_schedule):
            if current_step <= end_at:
                target_index = curret_index
                break
        res[i] = cond_schedule[target_index].cond

    return res


re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its assoicated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    Example:

        'a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).'

    produces:

    [
        ['a ', 1.0],
        ['house', 1.5730000000000004],
        [' ', 1.1],
        ['on', 1.0],
        [' a ', 1.1],
        ['hill', 0.55],
        [', sun, ', 1.1],
        ['sky', 1.4641000000000006],
        ['.', 1.1]
    ]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    return res
