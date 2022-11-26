select sessionid, 
    extract( second from max(event_timestamp) - min(event_timestamp) ) time_logged_in, 
    count(action_name) action_count,
    trim(to_char(min(event_timestamp), 'DAY')) login_day,
    max( regexp_replace(authentication_type,'.+HOST=([^\)]+).+', '\1') ) login_host,
    max( regexp_replace(authentication_type,'.+PORT=([^\)]+).+', '\1') ) login_port,
    max( dbusername ) db_username,
    max( os_username ) os_username,
    max( case when system_privilege_used is not null then 1 else 0 end ) system_privilege_used,
    max( case when return_code != 0 then 1 else 0 end ) non_success_return_code
from PROD_USER.UNIFIED_AUDIT_TRAIL_ARCHIVE
where sessionid in ( select unique sessionid 
                     from PROD_USER.UNIFIED_AUDIT_TRAIL_ARCHIVE 
                     where sessionid not in ( select sessionid from prod_user.sessions_summaries )
                     and rownum <= 100000)
group by sessionid;
