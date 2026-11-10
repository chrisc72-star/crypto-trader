-- SQLite
-- Look at the first 10 rows
SELECT * FROM candles LIMIT 10;

-- How many candles do you have?
SELECT COUNT(*) FROM candles;

-- Highest and lowest closing price ever
SELECT MAX(close), MIN(close) FROM candles;

-- The 10 biggest single-day gains, as a percentage
SELECT date, ROUND((close - open) / open * 100, 2) AS pct_change
FROM candles
ORDER BY pct_change DESC
LIMIT 10;

-- Average volume by year
SELECT SUBSTR(date, 1, 4) AS year, AVG(volume) AS avg_volume
FROM candles
GROUP BY year
ORDER BY year;