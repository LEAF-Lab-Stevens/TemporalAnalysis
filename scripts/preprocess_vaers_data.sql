BEGIN;
DROP TABLE IF EXISTS vaers_data;

CREATE TABLE vaers_data AS
SELECT v.vaers_id,
       v.vax_type,
       v.vax_manu,
       v.vax_name,
       d.age_yrs,
       d.sex,
       DATE_PART('year', d.recvdate::date) as vax_year,
       v.year,
       d.symptom_text,
       JSON_AGG(DISTINCT TRIM(spr.symptom)) AS symptom_list
FROM vax v
         INNER JOIN data d ON v.vaers_id = d.vaers_id
         INNER JOIN symptoms s ON v.vaers_id = s.vaers_id
         INNER JOIN (
    SELECT vaers_id, symptom1 AS symptom FROM symptoms WHERE symptom1 IS NOT NULL
    UNION ALL
    SELECT vaers_id, symptom2 AS symptom FROM symptoms WHERE symptom2 IS NOT NULL
    UNION ALL
    SELECT vaers_id, symptom3 AS symptom FROM symptoms WHERE symptom3 IS NOT NULL
    UNION ALL
    SELECT vaers_id, symptom4 AS symptom FROM symptoms WHERE symptom4 IS NOT NULL
    UNION ALL
    SELECT vaers_id, symptom5 AS symptom FROM symptoms WHERE symptom5 IS NOT NULL
) AS spr ON v.vaers_id = spr.vaers_id
WHERE v.vax_type ilike '%COVID19%'
GROUP BY v.vaers_id, v.vax_type, v.vax_manu, v.vax_name,
         d.age_yrs, d.sex, d.symptom_text,
         DATE_PART('year', d.recvdate::date), v.year;

END;