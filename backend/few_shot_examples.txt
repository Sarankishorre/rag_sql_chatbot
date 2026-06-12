# ── Few-shot examples ─────────────────────────────────────────────────────────
# These teach Groq the correct SQL patterns for YOUR database.
# Every time generate_sql() is called, these go into the system prompt.
# This fixes: LIMIT 1 bug, subquery bug, decimal vs percentage bug.

FEW_SHOT_EXAMPLES = """
Question: What is the death rate in each passenger class?
SQL:
SELECT 
    Pclass,
    COUNT(*) AS total_passengers,
    SUM(CASE WHEN Survived = 0 THEN 1 ELSE 0 END) AS total_deaths,
    ROUND(SUM(CASE WHEN Survived = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS death_rate_pct
FROM titanic
GROUP BY Pclass
ORDER BY Pclass
---
Question: What is the survival rate for each class?
SQL:
SELECT 
    Pclass,
    COUNT(*) AS total_passengers,
    ROUND(SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS survival_rate_pct
FROM titanic
GROUP BY Pclass
ORDER BY Pclass
---
Question: What is the death rate for each class broken down by gender?
SQL:
SELECT 
    Pclass,
    Sex,
    COUNT(*) AS total_passengers,
    ROUND(SUM(CASE WHEN Survived = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS death_rate_pct
FROM titanic
GROUP BY Pclass, Sex
ORDER BY Pclass, Sex
---
Question: How many passengers were in each class?
SQL:
SELECT 
    Pclass,
    COUNT(*) AS total_passengers
FROM titanic
GROUP BY Pclass
ORDER BY Pclass
---
Question: What is the survival rate for male vs female passengers?
SQL:
SELECT 
    Sex,
    COUNT(*) AS total_passengers,
    ROUND(SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS survival_rate_pct
FROM titanic
GROUP BY Sex
---
Question: Which embarked port had the worst survival rate?
SQL:
SELECT 
    Embarked,
    COUNT(*) AS total_passengers,
    ROUND(SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS survival_rate_pct
FROM titanic
WHERE Embarked IS NOT NULL
GROUP BY Embarked
ORDER BY survival_rate_pct ASC
---
Question: What is the average fare paid by each class?
SQL:
SELECT 
    Pclass,
    ROUND(AVG(Fare), 2) AS avg_fare
FROM titanic
GROUP BY Pclass
ORDER BY Pclass
---
Question: Show me passengers who paid more than 100 fare and survived?
SQL:
SELECT 
    PassengerId, Name, Sex, Age, Pclass, Fare
FROM titanic
WHERE Fare > 100 AND Survived = 1
ORDER BY Fare DESC
"""