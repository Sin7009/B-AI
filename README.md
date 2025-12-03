# B-AI

'''
graph TD
    %% Палитра (Sberbank approximates)
    classDef input fill:#FAFAF1,stroke:#333,stroke-width:1px,color:#000;
    classDef guard fill:#00703B,stroke:#fff,stroke-width:2px,color:#fff; 
    classDef strategy fill:#429538,stroke:#fff,stroke-width:1px,color:#fff;
    classDef tactic fill:#7CC344,stroke:#333,stroke-width:1px,color:#000;
    classDef feedback fill:#E3E6A1,stroke:#333,stroke-width:1px,color:#000;

    User(Вход: Проблема) --> ContextUnit{1. Анализ Контекста}

    subgraph "ZONE 1: GUARDRAILS (Фильтры Входа)"
        ContextUnit -- "Defcon 1 (Кризис)" --> Emergency[Аварийный Протокол]
        ContextUnit -- "Штатно" --> SocraticFW{2. Сократический Файервол}
        
        SocraticFW -- "Мало данных" --> QuestionGen(Генерация вопросов)
        QuestionGen -.-> User
        
        SocraticFW -- "Данные OK" --> BiasFilter{3. Психо-Фильтр}
        
        BiasFilter -- "Искажения/Эмоции" --> Therapist[Агент-Терапевт<br><i>Снятие тревоги</i>]
        BiasFilter -- "Манипуляция" --> Consigliere[Агент-Консильери<br><i>Оценка рисков</i>]
        BiasFilter -- "Чисто" --> Strategist
    end

    Therapist --> Strategist
    Consigliere --> Strategist

    subgraph "ZONE 2: STRATEGY CORE (Мозг)"
        Strategist(4. Методолог-Оркестратор) --> CultureCheck{Проверка Культуры}
        CultureCheck -- "Госсектор/Büro" --> SetRigid[Пресет: Консерватор]
        CultureCheck -- "Стартап/Tech" --> SetAgile[Пресет: Инноватор]
        
        SetRigid --> ParallelOps
        SetAgile --> ParallelOps
    end

    subgraph "ZONE 3: TACTICAL EXECUTION (Руки)"
        ParallelOps[5. Параллельные Агенты] 
        
        ParallelOps --> Ag_Sys[Системный Анализ]
        ParallelOps --> Ag_Cre[Креатив / ТРИЗ]
        ParallelOps --> Ag_Crit[Критик / Адвокат Дьявола]
        
        Ag_Sys -.-> FactCheck[Zero-Trust Checker<br><i>Web Search</i>]
        Ag_Cre -.-> FactCheck
        
        FactCheck -- "Факты подтверждены" --> Synthesizer(6. Синтез Решения)
        FactCheck -- "Фейк" --> ParallelOps
    end

    Synthesizer --> Output(7. Финальный Ответ)

    subgraph "ZONE 4: FEEDBACK LOOP (Обучение)"
        Output -.-> UserFeedback(Реакция Пользователя)
        UserFeedback -- "Не сработало" --> PostMortem[Анализ Провала]
        PostMortem --> Strategist
    end

    class User,QuestionGen,UserFeedback input;
    class ContextUnit,SocraticFW,BiasFilter,Therapist,Consigliere guard;
    class Strategist,CultureCheck,SetRigid,SetAgile strategy;
    class ParallelOps,Ag_Sys,Ag_Cre,Ag_Crit,Synthesizer tactic;
    class Emergency,FactCheck,PostMortem feedback;

'''
