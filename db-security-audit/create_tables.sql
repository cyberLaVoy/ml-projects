CREATE TABLE PROD_USER.SESSIONS_SUMMARIES (
    SESSIONID int primary key,
    TIME_LOGGED_IN float,
    ACTION_COUNT int,
    LOGIN_DAY varchar(255),
    LOGIN_HOST varchar(255),
    LOGIN_PORT int,
    DB_USERNAME varchar(255),
    OS_USERNAME varchar(255),
    SYSTEM_PRIVILEGE_USED int,
    NON_SUCCESS_RETURN_CODE int
);


CREATE TABLE PROD_USER.SESSIONS_ANOMALY_DETECTION (
    SESSIONID int primary key,
    CENTROID_DISTANCE float
);