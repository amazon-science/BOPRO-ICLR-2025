from utils.misc import stringify_grid

NEWLINE = "\n"


class Common:
    pass


class Semantle:
    @staticmethod
    def instruction(high_scoring, low_scoring=None, increasing_order=True, feedback=None,
                    target_score=1.0, exploit_threshold=None, prev_trials=None, repeats=None, task_demos=None,
                    scores=True, mix=False, no_demos=False):
        prompt = f"""Your task is to guess a hidden word from the English dictionary."""

        if not no_demos:
            prompt += f""" Use the below series of your \
previous guesses (in {"increasing" if increasing_order else "decreasing"} order of their similarity to the hidden \
word in terms of their *meaning*) to make a new guess. Your new guess should not have been made before"""
            if scores:
                prompt += " and should score higher than your previous guesses"
            prompt += "."

            if scores and target_score is not None:
                target_score = round(target_score, 4)
                prompt += f""" Analyze your previous guesses to decide what word could reach \
a target score of {target_score:.4f}."""
            else:
                prompt += " Analyze your previous guesses to decide what word to guess next."

            if scores and exploit_threshold is not None:
                exploit_threshold = round(exploit_threshold, 4)
                prompt += f""" If your best guesses are far (less than {exploit_threshold:.4f}), try making risky \
guesses that will help you explore new words, but if you're close (more than {exploit_threshold:.4f}), try making \
guesses that resemble (but are not identical) to the top words in *meaning*."""

        prompt += f""" If you guess an invalid word (i.e., not in the dictionary or a repeat guess), you will get no \
score, so stick to proper, single-word English words, and do not repeat your previous guesses!"""

        if repeats is not None:
            prompt += f"""In particular, DO NOT REPEAT the following words: \
{", ".join(map(lambda x: '"' + x.replace("%", "%%") + '"', repeats))}."""

        if not no_demos and low_scoring is not None:
            prompt += f"""\n
Here are some of your low-scoring previous guesses: 
{", ".join(map(lambda x: x.replace("%", "%%"), low_scoring))}"""

        if not no_demos and high_scoring is not None:
            prompt += f"""\n\nHere are your top previous guesses \
(from {["worst", "best"][0 if increasing_order else 1]} to {["worst", "best"][1 if increasing_order else 0]}):"""
            for h in high_scoring:
                prompt += f"""\n\nWord: {h[0].replace("%", "%%")}"""
                if scores:
                    prompt += f"""\nScore: {"{:.4f}".format(h[1])}"""

        if prev_trials is not None and len(prev_trials) > 0:
            prompt += f"""\nYour last {len(prev_trials)} guess(es) were:"""
            for prev in prev_trials:
                prompt += f"""\n\nWord: {prev[0].replace("%", "%%")}"""
                if scores:
                    prompt += f"""\nScore: {"{:.4f}".format(round(prev[-1], 4))}"""

        if scores and not no_demos:
            if target_score is not None:
                prompt += f"""
Now, guess exactly n=%s new word(s) that could give you a target score of {target_score:.4f}."""
            else:
                prompt += f"""
Now, guess exactly n=%s new word(s) that could improve your best score to reach the hidden word."""
        else:
            prompt += f"""
Now, guess exactly n=%s new word(s) that could be the hidden word. Be creative!"""

        prompt += f""" (Note: give only a list of word(s) in the provided JSON format, e.g. {{"response": ["word1", "word2",...]}})"""

        if feedback is not None:
            prompt += f"""{NEWLINE}{NEWLINE}Hint: {feedback}"""
        return prompt

    @staticmethod
    def warmstart(n_cands, previous=None, task_demos=None):
        prompt = f"""Your task is to guess a hidden test word from the English dictionary. \
Start by guessing {n_cands} diverse words to maximize your chance of guessing the hidden word. Be creative."""
        if previous is not None:
            prompt += f""" Do not guess any of the following words: {', '.join(previous)}."""
        return prompt

    @staticmethod
    def feedback(high_scoring, low_scoring=None, increasing_order=True, target_score=1.0, exploit_threshold=0.9,
                 prev_trials=None, task_demos=None):
        instruction = Semantle.instruction(high_scoring=high_scoring, low_scoring=low_scoring,
                                           increasing_order=increasing_order, target_score=target_score,
                                           exploit_threshold=exploit_threshold, prev_trials=prev_trials)
        feedback_prompts = {
            "generic": """Given the above task and the guesses so far, provide a promising, actionable strategy for me \
to guess the next word, but do not provide the actual word to guess. (Note: give the strategy as text in the provided \
JSON format)"""
        }
        prompt = f"""Read the following carefully:
{instruction}

{feedback_prompts["generic"]}"""
        return prompt


class ARC:
    @staticmethod
    def instruction(high_scoring, low_scoring=None, increasing_order=True, feedback=None,
                    target_score=1.0, exploit_threshold=None, prev_trials=None, repeats=None,
                    task_demos=None):
        prompt = f"""Given the following {len(task_demos)} examples of input-output grids, your task is to \
provide the underlying algorithm in natural language that transforms each input into its corresponding output. \
Here are the examples:

{(NEWLINE + NEWLINE).join(["INPUT #" + str(ti + 1) + ":" + NEWLINE + str(t[0]) + NEWLINE + "OUTPUT #" + str(ti + 1) + ":" + NEWLINE + str(t[1]) for ti, t in enumerate(task_demos)])}""" + \
                 (NEWLINE + NEWLINE)

        if target_score is not None:
            target_score = round(target_score, 4)
            prompt += f"""Use the below series of your previous guesses (in \
{"increasing" if increasing_order else "decreasing"} order of performance) to make a new guess that could \
reach a target score of {target_score:.4f}."""
        else:
            prompt += f"""Use the below series of your previous guesses (in \
{"increasing" if increasing_order else "decreasing"} order of performance) to make a new guess that will \
give you a score higher than any of them."""

        if exploit_threshold is not None:
            exploit_threshold = round(exploit_threshold, 4)
            prompt += f""" If your best guesses are far (less than {exploit_threshold:.4f}), try being more creative \
to explore solutions different from your previous guesses, but if you're close (more than {exploit_threshold:.4f}), \
make guesses that share some similarity with the top solutions but still improves them."""

        prompt += f""" Make sure that you do not repeat your previous guesses!"""

        if repeats is not None:
            prompt += f""" In particular, DO NOT REPEAT the following:
{(NEWLINE + NEWLINE).join(["REPEAT #" + str(ri + 1) + ":" + NEWLINE + r.replace("%", "%%") for ri, r in enumerate(repeats)])}"""

        prompt += (NEWLINE + NEWLINE)

        if low_scoring is not None:
            prompt += f"""Now, here are some of your low-scoring previous guesses:
{(NEWLINE + NEWLINE).join(["LOW-SCORING #" + str(li + 1) + ":" + NEWLINE + l.replace("%", "%%") for li, l in enumerate(low_scoring)])}""" + \
                      (NEWLINE + NEWLINE)

        prompt += f"""Here are your top previous guesses (from {["worst", "best"][0 if increasing_order else 1]} to {["worst", "best"][1 if increasing_order else 0]}):

{(NEWLINE + NEWLINE).join(["GUESS:" + NEWLINE + h[0].replace("%", "%%") + NEWLINE + "SCORE: " + "{:.4f}".format(h[1]) for h in high_scoring])}""" + \
                  (NEWLINE + NEWLINE)

        if prev_trials is not None and len(prev_trials) > 0:
            prompt += f"""Here are also your last {len(prev_trials)} previous guess{"es" if len(prev_trials) > 1 else ""}:
{(NEWLINE + NEWLINE).join(["GUESS:" + NEWLINE + p[0].replace("%", "%%") + NEWLINE + "SCORE: " + "{:.4f}".format(p[1]) for p in prev_trials])}""" + \
                      (NEWLINE + NEWLINE)

        if target_score is not None:
            prompt += f"""Now, make %s new guess that is likely to give you a target score of {target_score:.4f}. \
Be concise."""
        else:
            prompt += f"""Now, make %s new guess that is likely to improve your best score so far. Be concise."""

        prompt += f""" (Note: give only your guess as text in the provided JSON format)"""

        if feedback is not None:
            prompt += (NEWLINE + NEWLINE) + f"""HINT: {feedback}"""
        return prompt

    @staticmethod
    def warmstart(n_cands, task_demos=None, previous=None):
        prompt = f"""Given the following {len(task_demos)} examples of input-output grids, your task is to \
provide the underlying algorithm in natural language that transforms each input into its corresponding output. Here are the examples:

{(NEWLINE + NEWLINE).join(["INPUT #" + str(ti + 1) + ":" + NEWLINE + str(t[0]) + NEWLINE + "OUTPUT #" + str(ti + 1) + ":" + NEWLINE + str(t[1]) for ti, t in enumerate(task_demos)])}""" + \
                 (NEWLINE + NEWLINE)

        prompt += f"""Start by guessing {n_cands} diverse algorithms that are likely to maximize your chance of finding \
the correct solution. Be creative and concise.""" + (NEWLINE + NEWLINE)
        if previous is not None:
            prompt += f"""Do not repeat any of the following guesses:
{(NEWLINE + NEWLINE).join(["REPEAT #" + str(ri + 1) + ":" + NEWLINE + r.replace("%", "%%") for ri, r in enumerate(previous)])}""" + \
                      (NEWLINE + NEWLINE)

        prompt += f"""(Note: give only your guesses as text in the provided JSON format)"""
        return prompt

    @staticmethod
    def feedback(high_scoring, low_scoring=None, increasing_order=True, target_score=1.0, exploit_threshold=0.9,
                 prev_trials=None, task_demos=None):
        instruction = ARC.instruction(high_scoring=high_scoring, low_scoring=low_scoring,
                                      increasing_order=increasing_order, target_score=target_score,
                                      exploit_threshold=exploit_threshold, prev_trials=prev_trials,
                                      task_demos=task_demos)
        feedback_prompts = {
            "generic": """Given the above task and the guesses so far, provide a promising, actionable strategy for me \
to make the next guess, but do not provide the actual guess. (Note: give the strategy as text in the provided \
JSON format)"""
        }
        prompt = f"""Read the following carefully:
{instruction}

{feedback_prompts["generic"]}"""
        return prompt

    @staticmethod
    def blackbox(instruction, input):
        prompt = f"""Your task is to take as input a 2D grid of numbers and apply the provided transformation \
instructions to generate an output grid. The transformation instructions are as follows:

{instruction}

Here is the input grid:
{input}

What is the output grid? (Note: give only the output grid in the provided JSON format)"""

        return prompt


class ARCCode:
    #     prior_text = """The numbers in the grid are purely symbolic and may be treated as unique colors that form \
    # patterns or objects in the grids. The transformation could involve counting objects, comparing objects \
    # (e.g., which shape appears the most, which is the largest object, which objects are the same size), or repeating a \
    # pattern for a fixed number of times. Objects can be shapes like rectangles, triangles, and circles which can be \
    # mirrored, rotated, translated, deformed, combined, repeated, etc. Differences in distances can be detected. \
    # Cells with 0 may be treated as empty (black) cells."""
    prior_text = """Each number in a grid may be treated as a unique color and a 0 may be treated as an empty or black \
cell. Continuous sequences of numbers row-wise, column-wise, or diagonally may represent objects forming patterns. \
The transformation involves figuring out how these object patterns change from the input to the output. \
For e.g., object patterns may contain shapes like rectangles, triangles, or crosses which can then be \
mirrored, rotated, translated, deformed, combined, repeated, etc."""

    @staticmethod
    def instruction(high_scoring=None, low_scoring=None, increasing_order=True, generate_guess=True, feedback=None,
                    target_score=1.0, exploit_threshold=None, prev_trials=None, repeats=None,
                    task_demos=None, priors=True, use_numpy=True, transpose=False, docstring=True,
                    code=True, scores=True, mix=False, no_demos=False, output_docstring_only=False):
        prompt = f"""Given {len(task_demos) if task_demos is not None else "some"} examples of input-output grids of \
integers from 0-9, your task is to determine the transformation logic common to all the examples \
{"" if output_docstring_only else "and provide a python function "}that converts each input grid into its \
corresponding output grid."""

        if task_demos is not None:
            prompt += " Here are the examples:"
            for ti, t in enumerate(task_demos):
                prompt += f"""\n
INPUT #{ti + 1}:
{stringify_grid(t[0], pretty=False)}
OUTPUT #{ti + 1}:
{stringify_grid(t[1], pretty=False)}"""
                if transpose:
                    prompt += f"""\nHere is also a transposed view:
INPUT #{ti + 1} (transposed):
{stringify_grid(t[0], pretty=False, transpose=True)}
OUTPUT #{ti + 1} (transposed):
{stringify_grid(t[1], pretty=False, transpose=True)}"""

        prompt += (NEWLINE + NEWLINE)

        if priors:
            prompt += ARCCode.prior_text + (NEWLINE + NEWLINE)

        if not output_docstring_only:
            io_type = "list[list[int]]" if not use_numpy else "np.ndarray"
            prompt += f"""Your function should have the following signature:
```python
def transform(input: {io_type}) -> {io_type}:
    # Your code here
```
The following packages are already imported so do not repeat them: \
numpy as np and itertools.""" + (NEWLINE + NEWLINE)

        if not no_demos and low_scoring is not None:
            prompt += f"""Now, here are some of your low-scoring previous guesses:"""
            for li, l in enumerate(low_scoring):
                prompt += f"""\n\nLOW-SCORING #{li + 1}:"""
                if docstring:
                    prompt += f"""\n<docstring>{l[0].replace("%", "%%")}</docstring>"""
                if code:
                    prompt += f"""\n```python
{l[1].replace("%", "%%")}
```"""
            prompt += (NEWLINE + NEWLINE)

        if not no_demos and high_scoring is not None:
            prompt += "Here are your top previous guesses"
            if scores:
                prompt += f" and scores (in {'ascending' if increasing_order else 'descending'} order of accuracy \
compared to the target solution)"
            prompt += ":"
            for h in high_scoring:
                prompt += "\n\nGUESS" + (f" (score: {h[1]:.4f})" if scores else "") + ":"
                if docstring:
                    prompt += f"""\n<docstring>{h[0][0].replace("%", "%%")}</docstring>"""
                if code:
                    prompt += f"""\n```python
{h[0][1].replace("%", "%%")}
```"""
            prompt += (NEWLINE + NEWLINE)

        if prev_trials is not None and len(prev_trials) > 0:
            prompt += f"""Here are also your last {len(prev_trials)} previous guess{"es" if len(prev_trials) > 1 else ""}:"""
            for p in prev_trials:
                prompt += "\n\nLAST GUESS" + (f" (score: {p[1]:.4f})" if scores else "") + ":"
                if docstring:
                    prompt += f"""\n<docstring>{p[0][0].replace("%", "%%")}</docstring>"""
                if code:
                    prompt += f"""\n```python
{p[0][1].replace("%", "%%")}
```"""
            prompt += (NEWLINE + NEWLINE)

        if generate_guess:
            if not no_demos:
                if scores:
                    if target_score is not None:
                        target_score = round(target_score, 4)
                        prompt += f"""{"Analyze" if not mix else "Combine"} your previous guesses to \
make exactly n=%s new guess(es) likely to give you a target score of {target_score:.4f}. Be creative in your logic!"""
                    else:
                        prompt += f"""{"Analyze" if not mix else "Combine"} your previous guesses to \
make exactly n=%s new guess(es) likely to improve your best score so far. Be creative in your logic!"""
                else:
                    prompt += f"""{"Analyze" if not mix else "Combine"} your previous guesses to make \
exactly n=%s new guess(es) likely to improve your best guess so far. Be creative in your logic!"""

                if scores and exploit_threshold is not None:
                    exploit_threshold = round(exploit_threshold, 4)
                    prompt += f""" If your best guesses are low-performing (less than {exploit_threshold:.4f}), take more \
risks in exploring new solutions different from your previous guesses, but if your guesses are high-performing (more \
than {exploit_threshold:.4f}), make guesses that build upon your top solutions and improve them."""
            else:
                prompt += f"""Make exactly n=%s new guess(es) that could solve the task. Be creative in your logic!"""
            prompt += f""" Make sure to not repeat any of your previous guesses!"""

            if repeats is not None:
                prompt += f""" In particular, DO NOT REPEAT the following:"""
                for ri, r in enumerate(repeats):
                    prompt += f"""\n\nREPEAT #{ri + 1}:"""
                    if docstring:
                        prompt += f"""\n<docstring>{r[0].replace("%", "%%")}</docstring>"""
                    if code:
                        prompt += f"""\n```python
{r[1].replace("%", "%%")}
```"""
            prompt += (NEWLINE + NEWLINE) if prompt[-2:] != (NEWLINE + NEWLINE) else ""

            prompt += f"""(Note: directly start your response using the specified output format)"""

            if feedback is not None:
                prompt += (NEWLINE + NEWLINE) + f"""HINT: {feedback}"""
        return prompt

    @staticmethod
    def warmstart(n_cands, task_demos=None, previous=None, priors=True, use_numpy=True, transpose=False,
                  output_docstring_only=False):
        prompt = f"""Given the following {len(task_demos)} examples of input-output grids of integers from 0-9, \
your task is to determine the transformation logic common to all examples \
{"" if output_docstring_only else "and provide a python function "}that converts \
each input grid into its corresponding output grid. Here are the examples:"""

        for ti, t in enumerate(task_demos):
            prompt += f"""\n
INPUT #{ti + 1}:
{stringify_grid(t[0], pretty=False)}
OUTPUT #{ti + 1}:
{stringify_grid(t[1], pretty=False)}"""
            if transpose:
                prompt += f"""\nHere is also a transposed view:
INPUT #{ti + 1} (transposed):
{stringify_grid(t[0], pretty=False, transpose=True)}
OUTPUT #{ti + 1} (transposed):
{stringify_grid(t[1], pretty=False, transpose=True)}"""

        prompt += (NEWLINE + NEWLINE)

        if priors:
            prompt += ARCCode.prior_text + (NEWLINE + NEWLINE)

        if not output_docstring_only:
            io_type = "list[list[int]]" if not use_numpy else "np.ndarray"
            prompt += f"""Your functions should have the following signature:
```python
def transform(input: {io_type}) -> {io_type}:
    # Your code here
```
The following packages are already imported so do not repeat them: \
`numpy as np` and `itertools`""" + (NEWLINE + NEWLINE)

        prompt += f"""Start by guessing exactly {n_cands} diverse solutions that could solve the task. Make sure to \
not repeat any previous docstring{"" if output_docstring_only else " or code"}. Be creative in the logic of your \
proposals!""" + (NEWLINE + NEWLINE)

        if previous is not None:
            prompt += f"""Do not repeat any of the following guesses:"""
            for ri, r in enumerate(previous):
                prompt += f"""\n\nREPEAT #{ri + 1}:
<docstring>{r[0].replace("%", "%%")}</docstring>"""
                if not output_docstring_only:
                    prompt += f"""
```python
{r[1].replace("%", "%%")}
```"""
            prompt += (NEWLINE + NEWLINE)

        prompt += f"""(Note: directly start your response using the specified output format)"""

        return prompt

    @staticmethod
    def feedback(high_scoring, low_scoring=None, increasing_order=True, target_score=1.0, exploit_threshold=0.9,
                 prev_trials=None, task_demos=None, priors=True, use_numpy=True, transpose=False, docstring=True,
                 code=True, scores=True, mix=False):
        instruction = ARCCode.instruction(high_scoring=high_scoring, low_scoring=low_scoring,
                                          increasing_order=increasing_order, target_score=target_score,
                                          exploit_threshold=exploit_threshold, prev_trials=prev_trials,
                                          task_demos=task_demos, priors=priors, use_numpy=use_numpy,
                                          transpose=transpose, docstring=docstring, code=code, scores=scores,
                                          mix=mix)
        feedback_prompts = {
            "generic": """Given the above task and the guesses so far, provide a promising, actionable strategy for me \
to make the next guess, but do not provide the actual guess. (Note: give the strategy as text in the provided \
JSON format)"""
        }
        prompt = f"""Read the following task description carefully:
{instruction}

{feedback_prompts["generic"]}"""
        return prompt

    @staticmethod
    def fix(candidate, error_msg=None, task_demos=None, priors=True, transpose=False, use_numpy=True,
            predictions=None, improve=False):
        instruction = ARCCode.instruction(generate_guess=False, task_demos=None, priors=priors,
                                          transpose=transpose, use_numpy=use_numpy)
        prompt = instruction

        if not improve:
            # Fix mode
            prompt += f"""\
To solve this task, I came up with the following logic (described in a docstring) and program, but the code has one or \
more bugs that prevent it from producing a valid prediction on execution. Please fix the issues and provide exactly \
1 new fixed python program along with the same docstring."""
        else:
            # Improve mode
            prompt += f"""\
To solve this task, I came up with the following logic (described in a docstring) and program, but I think my code \
does not correctly follow the logic described in my docstring. Please fix any issues in my solution and provide \
exactly 1 new fixed python program along with the same docstring."""

        prompt += f""" Make sure that the code does exactly what the docstring says it should do. Here was my solution:

<docstring>{candidate[0].replace("%", "%%")}</docstring>
```python
{candidate[1].replace("%", "%%")}
```""" + (NEWLINE + NEWLINE)

        if error_msg is not None:
            prompt += f"""Here is the error message:
{error_msg}""" + (NEWLINE + NEWLINE)
        elif predictions is not None:
            prompt += f"""Here are the predicted grids output by my code:"""
            for pi, p in enumerate(predictions):
                prompt += f"""\n
INPUT #{pi + 1}:
{stringify_grid(task_demos[pi][0], pretty=False)}
EXPECTED OUTPUT #{pi + 1}:
{stringify_grid(task_demos[pi][1], pretty=False)}
PREDICTED OUTPUT #{pi + 1}:
{stringify_grid(p, pretty=False)}"""
                if transpose:
                    prompt += f"""\nHere is also a transposed view:
INPUT #{pi + 1} (transposed):
{stringify_grid(task_demos[pi][0], pretty=False, transpose=True)}
EXPECTED OUTPUT #{pi + 1} (transposed):
{stringify_grid(task_demos[pi][1], pretty=False, transpose=True)}
PREDICTED OUTPUT #{pi + 1} (transposed):
{stringify_grid(p, pretty=False, transpose=True)}"""
            prompt += (NEWLINE + NEWLINE)

        prompt += f"""(Note: directly start your response with the corrected solution in the specified output format \
and do not say anything else)"""

        return prompt


class MolOpt:
    @staticmethod
    def instruction(high_scoring, low_scoring=None, increasing_order=True, feedback=None,
                    target_score=1.0, exploit_threshold=None, prev_trials=None, repeats=None,
                    scores=True, no_demos=False, task_demos=None, **kwargs):
        prompt = f"""Your task is to find the optimal drug molecule that has both a high druglikeness (QED) \
as well as a strong binding affinity (vina) to the target protein. While both properties are important, \
binding affinity is twice as important as the druglikeness."""

        if task_demos is not None:
            prompt += f""" Here is the target protein: {task_demos}."""

        if not no_demos:
            prompt += f""" Use the below series of your \
previous guesses (in {"increasing" if increasing_order else "decreasing"} order of their scalarized scores \
(between 0 and 1) to make a new guess to maximize the score. Your new guess should not have been made before"""
            if scores:
                prompt += " and should score higher than your previous guesses"
            prompt += "."

            if scores and target_score is not None:
                target_score = round(target_score, 4)
                prompt += f""" Analyze your previous guesses to decide what molecule could reach \
the target score of {target_score:.4f}."""
            else:
                prompt += " Analyze your previous guesses to decide what molecule to guess next."

            if scores and exploit_threshold is not None:
                exploit_threshold = round(exploit_threshold, 4)
                prompt += f""" If your best guesses are far (less than {exploit_threshold:.4f}), try making risky \
guesses that will help you explore new molecules, but if you're close (more than {exploit_threshold:.4f}), try making \
guesses that resemble (but are not identical) to the top molecules."""

        prompt += f""" If you propose an invalid molecule or make a repeat guess, you will get no \
score, so stick to valid SMILES strings, and do not repeat your previous guesses!"""

        if repeats is not None:
            prompt += f"""In particular, DO NOT REPEAT the following molecules: \
{", ".join(map(lambda x: '"' + x.replace("%", "%%") + '"', repeats))}."""

        if not no_demos and low_scoring is not None:
            prompt += f"""\n
Here are some of your low-scoring previous guesses: 
{", ".join(map(lambda x: x.replace("%", "%%"), low_scoring))}"""

        if not no_demos and high_scoring is not None:
            prompt += f"""\n\nHere are your top previous guesses \
(from {["worst", "best"][0 if increasing_order else 1]} to {["worst", "best"][1 if increasing_order else 0]}):"""
            for h in high_scoring:
                prompt += f"""\n\nMolecule: {h[0].replace("%", "%%")}"""
                if scores:
                    prompt += f"""\nScore: {"{:.4f}".format(h[1])}"""

        if prev_trials is not None and len(prev_trials) > 0:
            prompt += f"""\nYour last {len(prev_trials)} guess(es) were:"""
            for prev in prev_trials:
                prompt += f"""\n\nMolecule: {prev[0].replace("%", "%%")}"""
                if scores:
                    prompt += f"""\nScore: {"{:.4f}".format(round(prev[-1], 4))}"""

        if scores and not no_demos:
            if target_score is not None:
                prompt += f"""
Now, guess exactly n=%s new molecule(s) that could give you a target score of {target_score:.4f}."""
            else:
                prompt += f"""
Now, guess exactly n=%s new molecule(s) that could improve your best score to reach the optimal molecule."""
        else:
            prompt += f"""
Now, guess exactly n=%s new molecule(s) that could be the optimal word. Be creative!"""

        prompt += f""" (Note: give only a list of SMILES strings in the provided JSON format, e.g. \
{{"response": ["SMILES1", "SMILES2", ...]}})"""

        if feedback is not None:
            prompt += f"""{NEWLINE}{NEWLINE}Hint: {feedback}"""
        return prompt

    @staticmethod
    def warmstart(n_cands, previous=None, task_demos=None, **kwargs):
        prompt = f"""Your task is to find the optimal drug molecule that has the highest druglikeness (QED) as well as \
the highest binding affinity (negative vina score) to the given protein."""

        if task_demos is not None:
            prompt += f""" Here is the target protein: {task_demos}."""

        prompt = f""" Start by guessing {n_cands} diverse molecules to maximize your chance of finding the optimal \
molecule. Be creative."""

        if previous is not None:
            prompt += f""" Do not guess any of the following molecules: {', '.join(previous)}."""

        prompt += f""" (Note: give only a list of SMILES strings in the provided JSON format, e.g. \
{{"response": ["SMILES1", "SMILES2", ...]}})"""

        return prompt

    @staticmethod
    def feedback(high_scoring, low_scoring=None, increasing_order=True, target_score=1.0, exploit_threshold=0.9,
                 prev_trials=None, task_demos=None, **kwargs):
        instruction = Semantle.instruction(high_scoring=high_scoring, low_scoring=low_scoring,
                                           increasing_order=increasing_order, target_score=target_score,
                                           exploit_threshold=exploit_threshold, prev_trials=prev_trials,
                                           task_demos=task_demos)
        feedback_prompts = {
            "generic": """Given the above task and the guesses so far, provide a promising, actionable strategy for me \
to guess the next molecule, but do not provide the actual molecule to guess. \
(Note: give the strategy as text in the provided JSON format)"""
        }
        prompt = f"""Read the following carefully:
{instruction}

{feedback_prompts["generic"]}"""
        return prompt
