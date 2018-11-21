CREATE TABLE Training_Info (
    TrainingId SERIAL PRIMARY KEY,
    TrainingDescription text,
    ModelLocation text NOT NULL,
    --Consider additional metadata like a path to zip file
    --containing the pipeline.config, model, etc.
    ModifiedDtim timestamp NOT NULL default current_timestamp,
    CreatedDtim timestamp NOT NULL default current_timestamp
);
