-- JV2AI (MySQL) から出走馬の血統を抽出するクエリ
--
-- 前提: JV-Data の SE(馬毎レース情報) と UM(競走馬マスタ) を JV2AI で取り込み済み.
--   出走馬テーブル  umagoto_race_joho : SE レコード (1行 = 1レース1頭)
--   競走馬マスタ    kyosoba_master2   : UM レコード (ketto1〜14_bamei / ketto1〜14_hanshoku_toroku_bango)
-- 血統関連の列名は STEP 0 の結果で確認済み. レース特定用の列 (kaisai_nen 等) は STEP 0b で確認.
-- UM の3代血統の並び順 (JV-Data仕様):
--   1:父 2:母 3:父父 4:父母 5:母父 6:母母 7:父父父 8:父父母
--   9:父母父 10:父母母 11:母父父 12:母父母 13:母母父 14:母母母

-- =====================================================================
-- STEP 0: 実テーブル/カラム名の確認 (血統登録番号・繁殖・馬名を含む列を探す)
-- =====================================================================
SELECT table_name, column_name, column_comment
FROM information_schema.columns
WHERE table_schema = DATABASE()
  AND (column_name    REGEXP 'ketto|hanshoku|bamei|血統|繁殖|馬名'
    OR column_comment REGEXP '血統|繁殖|馬名')
ORDER BY table_name, ordinal_position;

-- STEP 0b: umagoto_race_joho のレース特定用の列名を確認 (STEP 1-3 の kaisai_nen, kaisai_tsukihi,
--          keibajo_code, race_bango, wakuban, umaban と違う場合は置換する)
SELECT column_name
FROM information_schema.columns
WHERE table_schema = DATABASE() AND table_name = 'umagoto_race_joho'
ORDER BY ordinal_position;

-- =====================================================================
-- STEP 1: 抽出条件
-- =====================================================================
SET @kaisai_nen     = '2026';   -- 開催年 (YYYY)
SET @kaisai_tsukihi = '1004';   -- 開催月日 (MMDD)
SET @keibajo_code   = '06';     -- 競馬場コード (01札幌 02函館 03福島 04新潟 05東京 06中山 07中京 08京都 09阪神 10小倉)
SET @race_bango     = NULL;     -- レース番号 ('11' 等). NULL なら当日全レース

-- =====================================================================
-- STEP 2: 出走馬 × 3代血統
-- =====================================================================
SELECT
    se.kaisai_nen,
    se.kaisai_tsukihi,
    se.keibajo_code,
    se.race_bango,
    se.wakuban,
    se.umaban,
    se.ketto_toroku_bango,
    se.bamei,
    -- 父系・母系の主要どころ
    um.ketto1_bamei   AS 父,
    um.ketto2_bamei   AS 母,
    um.ketto5_bamei   AS 母父,
    um.ketto3_bamei   AS 父父,
    um.ketto4_bamei   AS 父母,
    um.ketto6_bamei   AS 母母,
    um.ketto13_bamei  AS 母母父,
    -- 3代目 (残り)
    um.ketto7_bamei   AS 父父父,
    um.ketto8_bamei   AS 父父母,
    um.ketto9_bamei   AS 父母父,
    um.ketto10_bamei  AS 父母母,
    um.ketto11_bamei  AS 母父父,
    um.ketto12_bamei  AS 母父母,
    um.ketto14_bamei  AS 母母母,
    -- 他テーブル (産駒成績・繁殖馬マスタ HN 等) と結合するための繁殖登録番号
    um.ketto1_hanshoku_toroku_bango AS 父_繁殖登録番号,
    um.ketto2_hanshoku_toroku_bango AS 母_繁殖登録番号,
    um.ketto5_hanshoku_toroku_bango AS 母父_繁殖登録番号
FROM umagoto_race_joho AS se
LEFT JOIN kyosoba_master2 AS um   -- マスタ未取込の馬も出走馬として残す
       ON um.ketto_toroku_bango = se.ketto_toroku_bango
WHERE se.kaisai_nen     = @kaisai_nen
  AND se.kaisai_tsukihi = @kaisai_tsukihi
  AND se.keibajo_code   = @keibajo_code
  AND (@race_bango IS NULL OR se.race_bango = @race_bango)
ORDER BY se.race_bango, se.umaban;

-- =====================================================================
-- STEP 3 (任意): 3代内のインブリード (同名馬が2回以上出現) を出走馬ごとに列挙
-- =====================================================================
WITH entry AS (
    -- um.* にも bamei 等があるため, 出走馬側の列は別名にして重複列エラーを避ける
    SELECT se.race_bango AS r_no, se.umaban AS u_no, se.bamei AS horse, um.*
    FROM umagoto_race_joho AS se
    JOIN kyosoba_master2 AS um ON um.ketto_toroku_bango = se.ketto_toroku_bango
    WHERE se.kaisai_nen     = @kaisai_nen
      AND se.kaisai_tsukihi = @kaisai_tsukihi
      AND se.keibajo_code   = @keibajo_code
      AND (@race_bango IS NULL OR se.race_bango = @race_bango)
),
anc AS (  -- 14頭分を縦持ちに展開 (母系の牝馬もクロス判定に含める)
              SELECT r_no, u_no, horse, ketto1_hanshoku_toroku_bango  AS hn, ketto1_bamei  AS anc_name FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto2_hanshoku_toroku_bango,  ketto2_bamei  FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto3_hanshoku_toroku_bango,  ketto3_bamei  FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto4_hanshoku_toroku_bango,  ketto4_bamei  FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto5_hanshoku_toroku_bango,  ketto5_bamei  FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto6_hanshoku_toroku_bango,  ketto6_bamei  FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto7_hanshoku_toroku_bango,  ketto7_bamei  FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto8_hanshoku_toroku_bango,  ketto8_bamei  FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto9_hanshoku_toroku_bango,  ketto9_bamei  FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto10_hanshoku_toroku_bango, ketto10_bamei FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto11_hanshoku_toroku_bango, ketto11_bamei FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto12_hanshoku_toroku_bango, ketto12_bamei FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto13_hanshoku_toroku_bango, ketto13_bamei FROM entry
    UNION ALL SELECT r_no, u_no, horse, ketto14_hanshoku_toroku_bango, ketto14_bamei FROM entry
)
SELECT r_no AS race_bango, u_no AS umaban, horse AS bamei, anc_name AS クロス馬, COUNT(*) AS 出現回数
FROM anc
WHERE hn IS NOT NULL AND TRIM(hn) NOT IN ('', '0000000000')
GROUP BY r_no, u_no, horse, hn, anc_name
HAVING COUNT(*) >= 2
ORDER BY r_no, u_no;
