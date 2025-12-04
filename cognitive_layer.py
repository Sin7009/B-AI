from enum import Enum

class ProblemType(Enum):
    DIAGNOSIS = "diagnosis"   # Для Критика, Системного аналитика и Синтезатора
    DESIGN = "design"         # Для ТРИЗ-агента (генерация решений)

class CognitiveScaffolder:
    """
    Внедряет структуры успешного мышления (Reasoning Structures) в системные промпты агентов.
    """

    def __init__(self):
        self._patterns = {
            # Протокол для аналитиков и проверяющих (Synthesis, Critic, System)
            ProblemType.DIAGNOSIS: (
                "\n=== COGNITIVE REASONING PROTOCOL (DIAGNOSIS) ===\n"
                "Прежде чем дать финальный ответ, проведи 'Internal Monologue' по шагам:\n"
                "1. [Strategy]: Какую цель преследует этот анализ? (Найти риск, объединить факты?)\n"
                "2. [Verification]: Проверь входные данные. Нет ли галлюцинаций или логических дыр?\n"
                "3. [Knowledge Alignment]: Сравни текущую ситуацию с известными паттернами отказов или успеха.\n"
                "4. [Coherence]: Убедись, что твой вывод логически вытекает из предпосылок.\n"
                "================================================\n"
                "ВАЖНО: В ответе выдай ТОЛЬКО результат (как указано в основной инструкции), "
                "но используй этот процесс мышления, чтобы сделать вывод пуленепробиваемым.\n"
            ),

            # Протокол для генераторов идей (TRIZ)
            ProblemType.DESIGN: (
                "\n=== COGNITIVE REASONING PROTOCOL (DESIGN) ===\n"
                "Прежде чем предложить решение, выполни ментальную работу:\n"
                "1. [Abstraction]: Забудь про детали. В чем корень противоречия?\n"
                "2. [Goal Management]: Каков идеальный конечный результат (ИКР)?\n"
                "3. [Decomposition]: Разбей проблему на части. Какую часть можно инвертировать или удалить?\n"
                "4. [Compositionality]: Собери решение заново из простых принципов.\n"
                "=============================================\n"
                "Используй это, чтобы выдать ОДНО сильное, нестандартное решение.\n"
            )
        }

    def enhance_prompt(self, base_prompt: str, problem_type: ProblemType) -> str:
        """Добавляет когнитивный каркас к системному промпту."""
        scaffold = self._patterns.get(problem_type, "")
        return f"{base_prompt}\n{scaffold}"
