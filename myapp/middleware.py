
from django.http import HttpResponseRedirect


class RedirectOldDomainMiddleware:
    OLD_DOMAIN = "stainaiviewer.azurewebsites.net"
    NEW_DOMAIN = "https://stainai.net"

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        host = request.get_host().split(":")[0].lower()

        if host == self.OLD_DOMAIN and request.method in ("GET", "HEAD"):

            # Preserve SSO compatibility during migration
            if request.path.startswith("/auth/"):
                return self.get_response(request)

            new_url = self.NEW_DOMAIN + request.get_full_path()
            return HttpResponseRedirect(new_url)

        return self.get_response(request)
