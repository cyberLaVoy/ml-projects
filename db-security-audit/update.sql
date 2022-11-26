select sessionid, 
    time_logged_in, 
    action_count,
    decode(login_day,
    'MONDAY', 1, 
    'TUESDAY', 2,
    'WEDNESDAY', 3,
    'THURSDAY', 4,
    'FRIDAY', 5,
    'SATURDAY', 6,
    'SUNDAY', 7) login_day,
    replace(login_host, '.', '') login_host,
    login_port,
    nvl(db_username, 'unknown') db_username,
    nvl(os_username, 'unknown') os_username,
    system_privilege_used,
    non_success_return_code
from PROD_USER.SESSIONS_SUMMARIES;