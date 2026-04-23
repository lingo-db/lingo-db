import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.Properties;

/**
 * Diagnostic probe: exercises PreparedStatement directly so the retry wrapper
 * in AvaticaStatement.executeInternal doesn't swallow the underlying cause.
 */
public class JdbcProbe {
    public static void main(String[] args) throws Exception {
        String host = args.length > 0 ? args[0] : "127.0.0.1";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 31337;
        String url = "jdbc:arrow-flight-sql://" + host + ":" + port + "/?useEncryption=false";
        System.out.println("[probe] connecting to " + url);

        Properties props = new Properties();
        try (Connection conn = DriverManager.getConnection(url, props)) {
            // PreparedStatement path avoids the retry wrapper around Statement.executeQuery.
            try (PreparedStatement ps = conn.prepareStatement("select 1 + 2 as x")) {
                System.out.println("[probe] prepare() OK: " + ps);
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        System.out.println("[probe] row: x=" + rs.getInt(1));
                    }
                }
            } catch (Throwable t) {
                System.out.println("[probe] TOP-LEVEL THROWABLE: " + t);
                Throwable c = t;
                int depth = 0;
                while (c != null && depth < 10) {
                    System.out.println("  cause[" + depth + "] " + c.getClass().getName() + ": " + c.getMessage());
                    c = c.getCause();
                    depth++;
                }
                t.printStackTrace(System.out);
            }
        }
    }
}
