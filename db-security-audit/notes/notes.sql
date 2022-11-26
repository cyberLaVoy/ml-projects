
-- found these data points not useful (reason unknown)
-- max( case when system_privilege_used is not null then 1 else 0 end ) system_privilege_used,
-- max( case when return_code != 0 then 1 else 0 end ) non_success_return_code


-- attempting to quantify every row 
select regexp_replace(authentication_type,'.+TYPE=\(([^\)]+).+', '\1') TYPE,
regexp_replace(authentication_type,'.+HOST=([^\)]+).+', '\1') HOST,
regexp_replace(authentication_type,'.+PROTOCOL=([^\)]+).+', '\1') PROTOCOL,
regexp_replace(authentication_type,'.+PORT=([^\)]+).+', '\1') PORT,
to_char(event_timestamp, 'MM') MONTH,
to_char(event_timestamp, 'DD') DAY,
to_char(event_timestamp, 'YYYY') YEAR,
to_char(event_timestamp, 'DAY') WEEKDAY,
to_char(event_timestamp, 'HH24:MI:SS') TIME,
sessionid, proxy_sessionid, os_username, userhost, terminal, dbid, 
dbusername, dbproxy_username, client_program_name, dblink_info, 
action_name, return_code, sql_text, sql_binds, new_schema, new_name
system_privilege_used, system_privilege, audit_option, object_privileges
role, target_user, excluded_user, excluded_schema, current_user
from PROD_USER.UNIFIED_AUDIT_TRAIL_ARCHIVE
where rownum <= 1000000 ;



-- quick random sampling
select * 
from (  select sessionid, 
        extract( second from max(event_timestamp) - min(event_timestamp) ) time_logged_in, 
        count(action_name) action_count,
        decode(trim(to_char(min(event_timestamp), 'DAY')),
        'MONDAY', 1, 
        'TUESDAY', 2,
        'WEDNESDAY', 3,
        'THURSDAY', 4,
        'FRIDAY', 5,
        'SATURDAY', 6,
        'SUNDAY', 7) login_day,
        replace(max( regexp_replace(authentication_type,'.+HOST=([^\)]+).+', '\1') ), '.', '') logon_host,
        max( regexp_replace(authentication_type,'.+PORT=([^\)]+).+', '\1') ) logon_port
        from PROD_USER.UNIFIED_AUDIT_TRAIL_ARCHIVE
        where rownum <= 100000
        group by sessionid
        order by dbms_random.value 
        )
where rownum <= 10000
order by action_count desc;