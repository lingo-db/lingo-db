set persist=1;
CREATE TABLE aka_name (
                          id integer NOT NULL,
                          person_id integer NOT NULL,
                          name text NOT NULL,
                          imdb_index character varying(12),
                          name_pcode_cf character varying(5),
                          name_pcode_nf character varying(5),
                          surname_pcode character varying(5),
                          md5sum character varying(32),
                          primary key(id)
);

CREATE TABLE aka_title (
                           id integer NOT NULL,
                           movie_id integer NOT NULL,
                           title text NOT NULL,
                           imdb_index character varying(12),
                           kind_id integer NOT NULL,
                           production_year integer,
                           phonetic_code character varying(5),
                           episode_of_id integer,
                           season_nr integer,
                           episode_nr integer,
                           note text,
                           md5sum character varying(32),
                          primary key(id)
);

CREATE TABLE cast_info (
                           id integer NOT NULL,
                           person_id integer NOT NULL,
                           movie_id integer NOT NULL,
                           person_role_id integer,
                           note text,
                           nr_order integer,
                           role_id integer NOT NULL,
                          primary key(id)
);

CREATE TABLE char_name (
                           id integer NOT NULL,
                           name text NOT NULL,
                           imdb_index character varying(12),
                           imdb_id integer,
                           name_pcode_nf character varying(5),
                           surname_pcode character varying(5),
                           md5sum character varying(32),
                          primary key(id)
);

CREATE TABLE comp_cast_type (
                                id integer NOT NULL,
                                kind character varying(32) NOT NULL,
                          primary key(id)
);

CREATE TABLE company_name (
                              id integer NOT NULL,
                              name text NOT NULL,
                              country_code character varying(255),
                              imdb_id integer,
                              name_pcode_nf character varying(5),
                              name_pcode_sf character varying(5),
                              md5sum character varying(32),
                          primary key(id)
);

CREATE TABLE company_type (
                              id integer NOT NULL,
                              kind character varying(32) NOT NULL,
                          primary key(id)
);

CREATE TABLE complete_cast (
                               id integer NOT NULL,
                               movie_id integer,
                               subject_id integer NOT NULL,
                               status_id integer NOT NULL,
                          primary key(id)
);

CREATE TABLE info_type (
                           id integer NOT NULL,
                           info character varying(32) NOT NULL,
                          primary key(id)
);

CREATE TABLE keyword (
                         id integer NOT NULL,
                         keyword text NOT NULL,
                         phonetic_code character varying(5),
                          primary key(id)
);

CREATE TABLE kind_type (
                           id integer NOT NULL,
                           kind character varying(15) NOT NULL,
                          primary key(id)
);

CREATE TABLE link_type (
                           id integer NOT NULL,
                           link character varying(32) NOT NULL,
                          primary key(id)
);

CREATE TABLE movie_companies (
                                 id integer NOT NULL,
                                 movie_id integer NOT NULL,
                                 company_id integer NOT NULL,
                                 company_type_id integer NOT NULL,
                                 note text,
                          primary key(id)
);

CREATE TABLE movie_info (
                            id integer NOT NULL,
                            movie_id integer NOT NULL,
                            info_type_id integer NOT NULL,
                            info text NOT NULL,
                            note text,
                          primary key(id)
);

CREATE TABLE movie_info_idx (
                                id integer NOT NULL,
                                movie_id integer NOT NULL,
                                info_type_id integer NOT NULL,
                                info text NOT NULL,
                                note text,
                          primary key(id)
);

CREATE TABLE movie_keyword (
                               id integer NOT NULL,
                               movie_id integer NOT NULL,
                               keyword_id integer NOT NULL,
                          primary key(id)
);

CREATE TABLE movie_link (
                            id integer NOT NULL,
                            movie_id integer NOT NULL,
                            linked_movie_id integer NOT NULL,
                            link_type_id integer NOT NULL,
                          primary key(id)
);

CREATE TABLE name (
                      id integer NOT NULL,
                      name text NOT NULL,
                      imdb_index character varying(12),
                      imdb_id integer,
                      gender character varying(1),
                      name_pcode_cf character varying(5),
                      name_pcode_nf character varying(5),
                      surname_pcode character varying(5),
                      md5sum character varying(32),
                          primary key(id)
);

CREATE TABLE person_info (
                             id integer NOT NULL,
                             person_id integer NOT NULL,
                             info_type_id integer NOT NULL,
                             info text NOT NULL,
                             note text,
                          primary key(id)
);

CREATE TABLE role_type (
                           id integer NOT NULL,
                           role character varying(32) NOT NULL,
                          primary key(id)
);

CREATE TABLE title (
                       id integer NOT NULL,
                       title text NOT NULL,
                       imdb_index character varying(12),
                       kind_id integer NOT NULL,
                       production_year integer,
                       imdb_id integer,
                       phonetic_code character varying(5),
                       episode_of_id integer,
                       season_nr integer,
                       episode_nr integer,
                       series_years character varying(49),
                       md5sum character varying(32),
                          primary key(id)
);

copy aka_name from 'aka_name.csv' csv escape '\';
copy aka_title from 'aka_title.csv' csv escape '\';
copy cast_info from 'cast_info.csv' csv escape '\';
copy char_name from 'char_name.csv' csv escape '\';
copy company_name from 'company_name.csv' csv escape '\';
copy company_type from 'company_type.csv' csv escape '\';
copy comp_cast_type from 'comp_cast_type.csv' csv escape '\';
copy complete_cast from 'complete_cast.csv' csv escape '\';
copy info_type from 'info_type.csv' csv escape '\';
copy keyword from 'keyword.csv' csv escape '\';
copy kind_type from 'kind_type.csv' csv escape '\';
copy link_type from 'link_type.csv' csv escape '\';
copy movie_companies from 'movie_companies.csv' csv escape '\';
copy movie_info from 'movie_info.csv' csv escape '\';
copy movie_info_idx from 'movie_info_idx.csv' csv escape '\';
copy movie_keyword from 'movie_keyword.csv' csv escape '\';
copy movie_link from 'movie_link.csv' csv escape '\';
copy name from 'name.csv' csv escape '\';
copy person_info from 'person_info.csv' csv escape '\';
copy role_type from 'role_type.csv' csv escape '\';
copy title from 'title.csv' csv escape '\';
