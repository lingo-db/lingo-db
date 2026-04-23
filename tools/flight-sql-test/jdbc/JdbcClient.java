import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.Properties;

/**
 * Exercises the LingoDB Flight SQL server via the Apache Arrow Flight SQL
 * JDBC driver. Mirrors the Python ADBC tests: a plain query, a count,
 * a join, and a couple of prepared statements with parameters.
 *
 *   javac -cp flight-sql-jdbc-driver-19.0.0.jar JdbcClient.java
 *   java  -cp flight-sql-jdbc-driver-19.0.0.jar:. JdbcClient grpc://127.0.0.1:31337
 */
public class JdbcClient {
    public static void main(String[] args) throws Exception {
        String host = args.length > 0 ? args[0] : "127.0.0.1";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 31337;
        String url = "jdbc:arrow-flight-sql://" + host + ":" + port + "/?useEncryption=false";
        System.out.println("[jdbc] connecting to " + url);

        Properties props = new Properties();
        try (Connection conn = DriverManager.getConnection(url, props)) {
            // 1. Simple literal query
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery("select 1 + 2 as x, 'hello' as greeting")) {
                rs.next();
                int x = rs.getInt("x");
                String greeting = rs.getString("greeting");
                System.out.println("[jdbc] literal => x=" + x + ", greeting=" + greeting);
                if (x != 3) throw new AssertionError("x != 3");
                if (!"hello".equals(greeting)) throw new AssertionError("greeting != hello");
            }

            // 2. Catalog query
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery("select count(*) from studenten")) {
                rs.next();
                long n = rs.getLong(1);
                System.out.println("[jdbc] count(studenten) = " + n);
                if (n != 8) throw new AssertionError("count != 8");
            }

            // 3. Prepared statement with an integer parameter
            try (PreparedStatement ps = conn.prepareStatement(
                     "select name, semester from studenten where matrnr = ?")) {
                ps.setLong(1, 26120);
                try (ResultSet rs = ps.executeQuery()) {
                    rs.next();
                    String name = rs.getString("name");
                    long sem = rs.getLong("semester");
                    System.out.println("[jdbc] prepared (matrnr=26120) => " + name + ", " + sem);
                    if (!"Fichte".equals(name)) throw new AssertionError("expected Fichte");
                    if (sem != 10) throw new AssertionError("expected 10");
                }

                // Re-execute with a different value on the same prepared statement
                ps.setLong(1, 29555);
                try (ResultSet rs = ps.executeQuery()) {
                    rs.next();
                    String name = rs.getString("name");
                    System.out.println("[jdbc] rebind (matrnr=29555) => " + name);
                    if (!"Feuerbach".equals(name)) throw new AssertionError("expected Feuerbach");
                }
            }

            // 4. Prepared statement with mixed parameter types
            try (PreparedStatement ps = conn.prepareStatement(
                     "select name from studenten where semester >= ? and name <> ?")) {
                ps.setInt(1, 8);
                ps.setString(2, "Fichte");
                try (ResultSet rs = ps.executeQuery()) {
                    java.util.List<String> names = new java.util.ArrayList<>();
                    while (rs.next()) names.add(rs.getString(1));
                    java.util.Collections.sort(names);
                    System.out.println("[jdbc] mixed-params => " + names);
                    java.util.List<String> expected = java.util.Arrays.asList(
                        "Aristoxenos", "Jonas", "Xenokrates");
                    if (!names.equals(expected))
                        throw new AssertionError(names + " != " + expected);
                }
            }

            // 5. Expected failure
            try (Statement st = conn.createStatement()) {
                try (ResultSet rs = st.executeQuery("select * from no_such_table")) {
                    rs.next();
                    throw new AssertionError("expected unknown-table query to fail");
                } catch (Exception ex) {
                    System.out.println("[jdbc] expected error surfaced: "
                                       + ex.getClass().getSimpleName() + ": "
                                       + ex.getMessage().split("\n")[0]);
                }
            }
        }
        System.out.println("[jdbc] all assertions passed");
    }
}
