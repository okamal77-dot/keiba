-- JV2AI 再取込後に血統が NULL になる原因の切り分け用
-- STEP 1 と同じ抽出条件を先に設定してから #A〜#D を順に実行する.
SET @kaisai_nen   = '2026';
SET @kaisai_gappi = '1004';
SET @keibajo_code = '05';   -- 実際に開催した競馬場コードに合わせる

-- #A 馬マスタ/出走馬系テーブルの一覧と件数・更新日時 (v1.2.5 でテーブル名が変わっていないか)
SELECT table_name, table_rows, create_time, update_time
FROM information_schema.tables
WHERE table_schema = DATABASE()
  AND (table_name LIKE 'kyosoba%' OR table_name LIKE 'umagoto%' OR table_name LIKE 'hanshoku%')
ORDER BY table_name;

-- #B 各テーブルの実件数と最終取込日時
SELECT 'umagoto_race_joho' AS tbl, COUNT(*) AS 件数, MAX(insert_timestamp) AS 最終追加,
       MAX(CONCAT(kaisai_nen, kaisai_gappi)) AS 最新開催日
FROM umagoto_race_joho
UNION ALL
SELECT 'kyosoba_master2', COUNT(*), MAX(insert_timestamp), NULL
FROM kyosoba_master2;

-- #C 指定日の出走馬が馬マスタに何頭ヒットするか (血統登録番号で結合)
SELECT se.keibajo_code,
       COUNT(DISTINCT se.ketto_toroku_bango)                       AS 出走馬,
       COUNT(DISTINCT um.ketto_toroku_bango)                       AS マスタ一致,
       COUNT(DISTINCT se.ketto_toroku_bango) - COUNT(DISTINCT um.ketto_toroku_bango) AS マスタ未登録
FROM umagoto_race_joho AS se
LEFT JOIN kyosoba_master2 AS um ON um.ketto_toroku_bango = se.ketto_toroku_bango
WHERE se.kaisai_nen = @kaisai_nen AND se.kaisai_gappi = @kaisai_gappi
GROUP BY se.keibajo_code;

-- #D マスタに無い出走馬の例 (番号の桁数・空白の違いがないかも確認)
SELECT se.keibajo_code, se.race_bango, se.umaban,
       TRIM(TRAILING '　' FROM se.bamei) AS bamei,
       CONCAT('[', se.ketto_toroku_bango, ']') AS 血統登録番号,
       LENGTH(se.ketto_toroku_bango) AS 桁数
FROM umagoto_race_joho AS se
LEFT JOIN kyosoba_master2 AS um ON um.ketto_toroku_bango = se.ketto_toroku_bango
WHERE se.kaisai_nen = @kaisai_nen AND se.kaisai_gappi = @kaisai_gappi
  AND um.ketto_toroku_bango IS NULL
ORDER BY se.keibajo_code, se.race_bango, se.umaban
LIMIT 20;
