-- Create a temporary table to populate the non-symptom values from csv file
CREATE TEMP TABLE non_symptoms (
    value VARCHAR
);

COPY non_symptoms(value)
    FROM '../data/non_symptom_list_version_2.csv'
    DELIMITER ','
    CSV HEADER;

-- Update the symptoms table to remove non-symptoms records
UPDATE symptoms
SET
    symptom1 = CASE WHEN symptom1 IN (SELECT value FROM non_symptoms) THEN NULL ELSE symptom1 END,
    symptom2 = CASE WHEN symptom2 IN (SELECT value FROM non_symptoms) THEN NULL ELSE symptom2 END,
    symptom3 = CASE WHEN symptom3 IN (SELECT value FROM non_symptoms) THEN NULL ELSE symptom3 END,
    symptom4 = CASE WHEN symptom4 IN (SELECT value FROM non_symptoms) THEN NULL ELSE symptom4 END,
    symptom5 = CASE WHEN symptom5 IN (SELECT value FROM non_symptoms) THEN NULL ELSE symptom5 END
WHERE
    symptom1 IN (SELECT value FROM non_symptoms)
   OR symptom2 IN (SELECT value FROM non_symptoms)
   OR symptom3 IN (SELECT value FROM non_symptoms)
   OR symptom4 IN (SELECT value FROM non_symptoms)
   OR symptom5 IN (SELECT value FROM non_symptoms);

-- Drop the temporary table
DROP TABLE IF EXISTS non_symptoms;
