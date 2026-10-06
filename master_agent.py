import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from repair_agents_runner import RepairAgentsRunner


@dataclass
class MasterVerdict:
    status: str
    selected_output: dict
    all_iterations: list

    @property
    def final_text(self):
        return self.selected_output.get("repaired_text", "")


class MasterAgent:
    def __init__(self, threshold=0.80, max_iterations=3):
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.runner = RepairAgentsRunner()

    # 1. Meaning Preservation Agent Score
    def calculate_meaning_preservation_score(self, original_text, repaired_text):
        return SequenceMatcher(None, original_text, repaired_text).ratio()

    # 2. Grammar-Optimized Agent Score
    def calculate_grammar_optimized_score(self, text):
        if not text:
            return 0.0

        score = 1.0

        if not text[0].isupper():
            score -= 0.2

        if not text.endswith((".", "!", "?")):
            score -= 0.2

        if "  " in text:
            score -= 0.1

        if re.search(r"\s+[,.!?]", text):
            score -= 0.1

        return max(score, 0.0)

    # 3. Concise-Fluent Agent Score
    def calculate_concise_fluent_score(self, text):
        if not text:
            return 0.0

        words = text.lower().split()

        if len(words) == 0:
            return 0.0

        repeated_count = 0

        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                repeated_count += 1

        score = 1 - (repeated_count / len(words))
        return max(score, 0.0)

    # 4. Sequence-Optimized Agent Score
    def calculate_sequence_optimized_score(self, original_text, repaired_text):
        original_words = original_text.split()
        repaired_words = repaired_text.split()

        return SequenceMatcher(None, original_words, repaired_words).ratio()

    # Final Score
    def calculate_final_score(self, original_text, repaired_text):
        meaning_score = self.calculate_meaning_preservation_score(
            original_text,
            repaired_text
        )

        grammar_score = self.calculate_grammar_optimized_score(
            repaired_text
        )

        concise_fluent_score = self.calculate_concise_fluent_score(
            repaired_text
        )

        sequence_score = self.calculate_sequence_optimized_score(
            original_text,
            repaired_text
        )

        final_score = (
            0.50 * meaning_score +
            0.30 * grammar_score +
            0.15 * concise_fluent_score +
            0.05 * sequence_score
        )

        return {
            "meaning_preservation_score": meaning_score,
            "grammar_optimized_score": grammar_score,
            "concise_fluent_score": concise_fluent_score,
            "sequence_optimized_score": sequence_score,
            "final_score": final_score
        }

    # Evaluate all 4 agent outputs
    def evaluate_outputs(self, agent_outputs):
        scored_outputs = []

        for output in agent_outputs:
            original_text = output["original_text"]
            repaired_text = output["repaired_text"]

            scores = self.calculate_final_score(
                original_text,
                repaired_text
            )

            scored_outputs.append({
                "agent_name": output["agent_name"],
                "original_text": original_text,
                "repaired_text": repaired_text,
                "scores": scores
            })

        return scored_outputs

    # Select best output using threshold
    def select_best_output(self, scored_outputs):
        above_threshold = []

        for output in scored_outputs:
            if output["scores"]["final_score"] >= self.threshold:
                above_threshold.append(output)

        if len(above_threshold) == 1:
            return above_threshold[0]

        if len(above_threshold) > 1:
            return max(
                above_threshold,
                key=lambda x: x["scores"]["final_score"]
            )

        return None

    # Feedback loop
    def run(self, tokens, prompt=None):
        current_tokens = tokens
        all_iterations = []

        for iteration in range(1, self.max_iterations + 1):
            agent_outputs = self.runner.run_parallel(
                current_tokens,
                prompt
            )

            scored_outputs = self.evaluate_outputs(agent_outputs)

            all_iterations.append({
                "iteration": iteration,
                "scored_outputs": scored_outputs
            })

            best_output = self.select_best_output(scored_outputs)

            if best_output is not None:
                return MasterVerdict(
                    status="success",
                    selected_output=best_output,
                    all_iterations=all_iterations,
                )

            highest_output = max(
                scored_outputs,
                key=lambda x: x["scores"]["final_score"]
            )

            current_tokens = highest_output["repaired_text"].split()

        final_best = max(
            all_iterations[-1]["scored_outputs"],
            key=lambda x: x["scores"]["final_score"]
        )

        return MasterVerdict(
            status="threshold_not_met",
            selected_output=final_best,
            all_iterations=all_iterations,
        )
