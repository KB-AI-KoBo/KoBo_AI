package com.kb.kobo.configuration;

import org.springframework.boot.web.embedded.tomcat.TomcatServletWebServerFactory;
import org.springframework.boot.web.server.MimeMappings;
import org.springframework.boot.web.server.WebServerFactoryCustomizer;
import org.springframework.stereotype.Component;

@Component
public class CustomServletWebServerFactory implements WebServerFactoryCustomizer<TomcatServletWebServerFactory> {

    @Override
    public void customize(TomcatServletWebServerFactory factory) {
        MimeMappings mappings = new MimeMappings(MimeMappings.DEFAULT);
        mappings.add("static/css", "text/css");
        mappings.add("static/js", "application/javascript");
        mappings.add("json", "application/json");
        mappings.add("png", "image/png");
        mappings.add("jpg", "image/jpeg");
        mappings.add("jpeg", "image/jpeg");
        mappings.add("svg", "image/svg+xml");
        factory.setMimeMappings(mappings);
    }
}

