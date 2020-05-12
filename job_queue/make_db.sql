CREATE TABLE "jobs"
(
	infile varchar not null,
	outfile varchar not null,
	weightfile varchar not null,
	created_dt datetime default CURRENT_TIMESTAMP,
	pk integer
		constraint jobs_pk
			primary key autoincrement,
	status TEXT default 'scheduled' not null,
	check (status IN ('scheduled', 'complete', 'failed'))
)

