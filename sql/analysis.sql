SELECT * FROM customers WHERE Exited = 1;

SELECT COUNT(*) AS churn_count FROM customers WHERE Exited = 1;

SELECT Geography, COUNT(*) AS churn_count
FROM customers
WHERE Exited = 1
GROUP BY Geography
ORDER BY churn_count DESC;

SELECT Geography,
       COUNT(*)       AS total,
       SUM(Exited)    AS churn_count,
       ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY Geography
ORDER BY churn_rate_pct DESC;

SELECT CASE
         WHEN Age < 30 THEN '18-29'
         WHEN Age < 40 THEN '30-39'
         WHEN Age < 50 THEN '40-49'
         WHEN Age < 60 THEN '50-59'
         ELSE '60+'
       END AS age_group,
       COUNT(*) AS customer_count,
       ROUND(AVG(Balance), 2) AS avg_balance
FROM customers
GROUP BY age_group
ORDER BY age_group;

SELECT CASE WHEN IsActiveMember = 1 THEN 'active'
            ELSE 'inactive' END AS member_status,
       COUNT(*) AS total,
       ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY IsActiveMember
ORDER BY IsActiveMember;
