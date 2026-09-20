-- Runs once, automatically, when the postgres container first initializes
-- an empty data directory (official postgres image convention:
-- /docker-entrypoint-initdb.d/*.sql). Airflow's containers point at this
-- database (AIRFLOW__DATABASE__SQL_ALCHEMY_CONN), but Postgres only ever
-- auto-creates the POSTGRES_DB database (financeai) — not this one.
CREATE DATABASE airflow OWNER finai;
