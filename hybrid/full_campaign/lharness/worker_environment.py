"""One configured route for newly launched native workers; no global mutation."""
import os

PROXY_KEYS = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')
LOCAL_NO_PROXY = 'localhost,127.0.0.1,::1'


def worker_environment(config, base=None, **overrides):
    env = dict(os.environ if base is None else base)
    env.update(overrides)
    proxy = config.get('coding_agent', {}).get('proxy_url')
    if proxy:
        if not isinstance(proxy, str):
            raise ValueError('Configured coding proxy must be a string')
        env.update({key: proxy for key in PROXY_KEYS})
        env['NO_PROXY'] = env['no_proxy'] = LOCAL_NO_PROXY
    return env
