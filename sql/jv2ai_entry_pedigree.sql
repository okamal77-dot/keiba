-- JV2AI (MySQL) から出走馬の血統を抽出するクエリ
--
-- 前提: JV-Data の SE(馬毎レース情報) と UM(競走馬マスタ) を JV2AI で取り込み済み.
--   出走馬テーブル  race_uma : SE レコード (1行 = 1レース1頭)
--   競走馬マスタ    uma      : UM レコード (3代血統 14頭分の繁殖登録番号・馬名を持つ)
-- テーブル名・カラム名は JV2AI の実際の定義に合わせて置換すること (STEP 0 で確認).
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
    um.ketto3_bamei_1   AS 父,
    um.ketto3_bamei_2   AS 母,
    um.ketto3_bamei_5   AS 母父,
    um.ketto3_bamei_3   AS 父父,
    um.ketto3_bamei_4   AS 父母,
    um.ketto3_bamei_6   AS 母母,
    um.ketto3_bamei_13  AS 母母父,
    -- 3代目 (残り)
    um.ketto3_bamei_7   AS 父父父,
    um.ketto3_bamei_8   AS 父父母,
    um.ketto3_bamei_9   AS 父母父,
    um.ketto3_bamei_10  AS 父母母,
    um.ketto3_bamei_11  AS 母父父,
    um.ketto3_bamei_12  AS 母父母,
    um.ketto3_bamei_14  AS 母母母,
    -- 他テーブル (産駒成績・繁殖馬マスタ HN 等) と結合するための繁殖登録番号
    um.ketto3_hanshoku_toroku_bango_1 AS 父_繁殖登録番号,
    um.ketto3_hanshoku_toroku_bango_2 AS 母_繁殖登録番号,
    um.ketto3_hanshoku_toroku_bango_5 AS 母父_繁殖登録番号
FROM race_uma AS se
LEFT JOIN uma AS um                      -- マスタ未取込の馬も出走馬として残す
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
    SELECT se.race_bango, se.umaban, se.bamei, um.*
    FROM race_uma AS se
    JOIN uma AS um ON um.ketto_toroku_bango = se.ketto_toroku_bango
    WHERE se.kaisai_nen     = @kaisai_nen
      AND se.kaisai_tsukihi = @kaisai_tsukihi
      AND se.keibajo_code   = @keibajo_code
      AND (@race_bango IS NULL OR se.race_bango = @race_bango)
),
anc AS (  -- 14頭分を縦持ちに展開 (母系の牝馬もクロス判定に含める)
              SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_1  AS hn, ketto3_bamei_1  AS anc_name FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_2,  ketto3_bamei_2  FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_3,  ketto3_bamei_3  FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_4,  ketto3_bamei_4  FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_5,  ketto3_bamei_5  FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_6,  ketto3_bamei_6  FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_7,  ketto3_bamei_7  FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_8,  ketto3_bamei_8  FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_9,  ketto3_bamei_9  FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_10, ketto3_bamei_10 FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_11, ketto3_bamei_11 FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_12, ketto3_bamei_12 FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_13, ketto3_bamei_13 FROM entry
    UNION ALL SELECT race_bango, umaban, bamei, ketto3_hanshoku_toroku_bango_14, ketto3_bamei_14 FROM entry
)
SELECT race_bango, umaban, bamei, anc_name AS クロス馬, COUNT(*) AS 出現回数
FROM anc
WHERE hn IS NOT NULL AND TRIM(hn) NOT IN ('', '0000000000')
GROUP BY race_bango, umaban, bamei, hn, anc_name
HAVING COUNT(*) >= 2
ORDER BY race_bango, umaban;
