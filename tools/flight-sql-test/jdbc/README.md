# JDBC Flight SQL test client

Exercises the LingoDB Flight SQL server via the Apache Arrow Flight SQL JDBC
driver 19.0.0. Matches the Python ADBC tests (literal query, `count(*)`,
prepared statements with int/string parameters).

## Requirements

- OpenJDK 21 (or any JDK 11+). On Ubuntu: `sudo apt install openjdk-21-jdk-headless`.
- The LingoDB Flight SQL server binary (`build/lingodb-release/flight-sql-server`).

## Fetch the driver

```bash
curl -sSfL -o flight-sql-jdbc-driver-19.0.0.jar \
    https://repo1.maven.org/maven2/org/apache/arrow/flight-sql-jdbc-driver/19.0.0/flight-sql-jdbc-driver-19.0.0.jar
```

The JAR (~46 MB) is gitignored so each developer downloads a fresh copy.

## Run the tests

```bash
# 1) Start the server against the bundled 'uni' dataset
./build/lingodb-release/flight-sql-server resources/data/uni \
    --host 127.0.0.1 --port 31337

# 2) In another shell
cd tools/flight-sql-test/jdbc
javac -cp flight-sql-jdbc-driver-19.0.0.jar JdbcClient.java

java \
  --add-opens=java.base/java.nio=ALL-UNNAMED \
  --add-opens=java.base/java.lang=ALL-UNNAMED \
  --add-opens=java.base/java.util=ALL-UNNAMED \
  --add-opens=java.base/sun.nio.ch=ALL-UNNAMED \
  --add-opens=java.base/jdk.internal.misc=ALL-UNNAMED \
  -Dio.netty.tryReflectionSetAccessible=true \
  -cp flight-sql-jdbc-driver-19.0.0.jar:. JdbcClient 127.0.0.1 31337
```

The JVM module-access flags are required by the driver's bundled Netty to
access `sun.misc.Unsafe` on JDK 17+ — they're not LingoDB-specific.

## Files

- `JdbcClient.java` — full test suite (literal, count, prepared with rebind, mixed-param, expected-failure).
- `JdbcProbe.java` — diagnostic helper that uses `PreparedStatement` directly, bypassing Avatica's swallow-and-retry wrapper. Useful when Avatica logs only `"Failed after 5 attempts"` and you need the real cause.
