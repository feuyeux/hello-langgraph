package org.feuyeux.ai.hello;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@Slf4j
public class LanggraphService {
  @PostConstruct
  public void init() {
    log.info("LanggraphService init");
  }
}
