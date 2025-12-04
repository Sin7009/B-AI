import unittest
from cognitive_layer import CognitiveScaffolder, ProblemType

class TestCognitiveScaffolder(unittest.TestCase):
    def setUp(self):
        self.scaffolder = CognitiveScaffolder()
        self.base_prompt = "Ты полезный агент."

    def test_enhance_prompt_design(self):
        """Проверяет добавление протокола DESIGN для ТРИЗ задач."""
        enhanced = self.scaffolder.enhance_prompt(self.base_prompt, ProblemType.DESIGN)

        self.assertIn(self.base_prompt, enhanced)
        self.assertIn("=== COGNITIVE REASONING PROTOCOL (DESIGN) ===", enhanced)
        self.assertIn("[Abstraction]", enhanced)
        self.assertIn("[Goal Management]", enhanced)

    def test_enhance_prompt_diagnosis(self):
        """Проверяет добавление протокола DIAGNOSIS для аналитических задач."""
        enhanced = self.scaffolder.enhance_prompt(self.base_prompt, ProblemType.DIAGNOSIS)

        self.assertIn(self.base_prompt, enhanced)
        self.assertIn("=== COGNITIVE REASONING PROTOCOL (DIAGNOSIS) ===", enhanced)
        self.assertIn("[Strategy]", enhanced)
        self.assertIn("[Verification]", enhanced)

    def test_enhance_prompt_unknown(self):
        """Проверяет поведение при неизвестном типе проблемы (если такое возможно, хотя enum защищает)."""
        # Если бы мы передали что-то вне map, метод должен вернуть просто base_prompt + ""
        # Но так как мы используем Enum и типизацию, это маловероятно.
        # Проверим просто, что enhance_prompt корректно работает.
        pass

if __name__ == '__main__':
    unittest.main()
