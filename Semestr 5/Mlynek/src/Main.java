import java.util.*;
import sac.*;
import sac.game.*;
import sac.graph.AStar;

import javax.annotation.processing.SupportedSourceVersion;



class Mill extends GameStateImpl {

    // maximizingTurnNow == false => ruch bialych
    // maximizingTurnNow == true => ruch czarnych
    public byte[] board;
    public int pawnsToPlace;
    public int blackPawnsCnt = 0;
    public int whitePawnsCnt =0;

    private int[][] neighbourList = {
            {1, 7}, // 0
            {0, 2, 9}, // 1
            {1, 3}, // 2
            {2, 4, 11}, // 3
            {3, 5}, // 4
            {4, 6, 13}, // 5
            {5, 7}, // 6
            {6, 0, 15}, // 7
            {9, 15}, // 8
            {1, 8, 10, 17}, // 9
            {9, 11}, // 10
            {3, 10, 12, 19}, // 11
            {11, 13}, // 12
            {5, 12, 14, 21}, // 13
            {13, 15}, // 14
            {7, 14, 23, 8}, // 15
            {17, 23}, // 16
            {9, 16, 18}, // 17
            {17, 19}, // 18
            {11, 18, 20}, // 19
            {19, 21}, // 20
            {20, 22, 13}, // 21
            {21, 23}, // 22
            {15, 16, 22} // 23
    };

    public int[][] mills = {
            {0,1,2},{2,3,4},{4,5,6},{6,7,0},
            {8,9,10},{10,11,12},{12,13,14},{14,15,8},
            {16,17,18},{18,19,20},{20,21,22},{22,23,16},
            {3,11,19},{5,13,21},{1,9,17},{7,15,23}
    };

    public List<Integer> getJumps(){
        List<Integer> jumps = new ArrayList<>();
        for(int i=0;i<24;i++){
            if(board[i] == 0){
                jumps.add(i);
            }
        }
        return jumps;
    }

    public Mill() {
        this.board = new byte[24];
        for (int i = 0; i < 24; i++) {
            board[i] = 0;
        }
        this.pawnsToPlace = 18;
        this.maximizingTurnNow = true;
    }

    public Mill(Mill parent) {
        board = new byte[24];
        for (int i = 0; i < 24; i++) {
            board[i] = parent.board[i];
        }
        whitePawnsCnt = parent.whitePawnsCnt;
        blackPawnsCnt = parent.blackPawnsCnt;
        pawnsToPlace = parent.pawnsToPlace;
        maximizingTurnNow = parent.maximizingTurnNow;
    }

    @Override
    public List<GameState> generateChildren() {
        List<GameState> children = new ArrayList<>();
        if (pawnsToPlace > 0) {
            children.addAll(firstPhase());
        } else if (whitePawnsCnt == 3 && maximizingTurnNow || blackPawnsCnt == 3 && !maximizingTurnNow) {
            children.addAll(thirdPhase());
        } else if (whitePawnsCnt > 3 && maximizingTurnNow || blackPawnsCnt > 3 && !maximizingTurnNow) {
            children.addAll(secondPhase());
        }
//        System.out.println("Generated " + children.size() + " children for state: \n" + this.toString());
        return children;
    }

    @Override
    public int hashCode() {
        return toString().hashCode();
    }

    private List<GameState> firstPhase() {
        List<GameState> children = new ArrayList<>();
        for (int i = 0; i < 24; i++) {
            if (board[i] == 0) {
                Mill child = new Mill(this);
                child.board[i] = (byte) (maximizingTurnNow ? 'W' : 'B');
                child.pawnsToPlace--;
                if(!maximizingTurnNow){
                    child.blackPawnsCnt++;
                }else {
                    child.whitePawnsCnt++;
                }
                child.setMoveName(Integer.toString(i));
                if (child.isMillFormed(i)) {
                    List<Mill> millSolved = child.solveMill();
                    children.addAll(millSolved);
                    for(Mill c : millSolved){
                        c.setMoveName(i + " " + c.getMoveName());
                    }
                } else {
                    child.setMoveName(Integer.toString(i));
                    child.maximizingTurnNow = !child.maximizingTurnNow;
                    children.add(child);
                }
            }
        }
        return children;
    }

    private List<GameState> secondPhase() {
        List<GameState> children = new ArrayList<>();
        byte currentPlayer = (byte) (maximizingTurnNow ? 'W' : 'B');
        if(maximizingTurnNow){
            assert (whitePawnsCnt > 3);
        }else {
            assert (blackPawnsCnt > 3);
        }
        for(int i=0; i<24; i++){
            if(board[i] == currentPlayer){
                for(int neighbour : neighbourList[i]){
                    if(board[neighbour] == 0){
                        Mill child = new Mill(this);
                        child.board[i] = 0;
                        child.board[neighbour] = currentPlayer;
                        if(child.isMillFormed(neighbour)){
                            List<Mill> millSolved = child.solveMill();
                            for(Mill c : millSolved){
                                c.setMoveName(i + " " + neighbour + " " + c.getMoveName());
                            }
                            children.addAll(millSolved);
                        }else {
                            child.setMoveName(i + " " + neighbour);
                            child.maximizingTurnNow = !child.maximizingTurnNow;
                            children.add(child);
                        }
                    }
                }
            }
        }
        return children;
    }

    private List<GameState> thirdPhase(){
        List<GameState> children = new ArrayList<>();
        byte currentPlayer = (byte) (maximizingTurnNow ? 'W' : 'B');
        if(maximizingTurnNow){
            assert (whitePawnsCnt == 3);
        }else {
            assert (blackPawnsCnt == 3);
        }
        for(int i=0;i<24; i++){
            if(board[i] == currentPlayer){
                for(int j=0; j<24; j++){
                    if(board[j] == 0){
                        Mill child = new Mill(this);
                        child.board[i] = 0;
                        child.board[j] = currentPlayer;
                        if(child.isMillFormed(j)){
                            List<Mill> millSolved = child.solveMill();
                            for(Mill c : millSolved){
                                c.setMoveName(i + " " + j + " " + c.getMoveName());
                            }
                            children.addAll(millSolved);
                        }else {
                            child.setMoveName(i + " " + j);
                            child.maximizingTurnNow = !child.maximizingTurnNow;
                            children.add(child);
                        }
                    }
                }
            }
        }
        return children;
    }

    public boolean isMillFormed(int idx) {
        if (board[idx] == 0) return false;
        byte currentPlayer = (byte) (maximizingTurnNow ? 'W' : 'B');
        for (int[] mill : mills) {
            if (mill[0] == idx || mill[1] == idx || mill[2] == idx) {
                if (board[mill[0]] == currentPlayer && board[mill[1]] == currentPlayer && board[mill[2]] == currentPlayer) {
//                    System.out.println(mill[0] + " " + mill[1] + " " + mill[2]);
                    return true;
                }
            }
        }
        return false;
    }

    public List<Mill> solveMill() {
        List<Mill> children = new ArrayList<>();
        byte opponent = (byte) (maximizingTurnNow ? 'B' : 'W');
        boolean isRemoved = false;
        for (int i = 0; i < board.length; i++) {
            if (board[i] == opponent && !isMillFormed(i)) {
                Mill child = new Mill(this);
                child.board[i] = 0;
                isRemoved = true;
                if(maximizingTurnNow){
                    child.blackPawnsCnt--;
                }else {
                    child.whitePawnsCnt--;
                }
                child.maximizingTurnNow = !this.maximizingTurnNow;
                child.setMoveName(Integer.toString(i));
                children.add(child);
            }
        }
        if(!isRemoved){
            for (int i = 0; i < board.length; i++) {
                if (board[i] == opponent) {
                    Mill child = new Mill(this);
                    child.board[i] = 0;
                    if(maximizingTurnNow){
                        child.blackPawnsCnt--;
                    }else {
                        child.whitePawnsCnt--;
                    }
                    child.maximizingTurnNow = !this.maximizingTurnNow;
                    child.setMoveName(Integer.toString(i));
                    children.add(child);
                }
            }
        }
        return children;
    }

    @Override
    public String toString() {
        char[] displayBoard = new char[24];
        for (int i = 0; i < 24; i++) {
            if (board[i] == 0) {
                displayBoard[i] = '.';
            } else if (board[i] == 'W') {
                displayBoard[i] = 'W';
            } else {
                displayBoard[i] = 'B';
            }
        }

        return " " + displayBoard[6] + "--------" + displayBoard[5] + "--------" + displayBoard[4] + "\n" +
                " |        |        |\n" +
                " |  " + displayBoard[14] + "-----" + displayBoard[13] + "-----" + displayBoard[12] + "  |\n" +
                " |  |     |     |  |\n" +
                " |  |  " + displayBoard[22] + "--" + displayBoard[21] + "--" + displayBoard[20] + "  |  |\n" +
                " |  |  |     |  |  |\n" +
                displayBoard[7] + "--" + displayBoard[15] + "--" + displayBoard[23] + "     " + displayBoard[19] + "--" + displayBoard[11] + "--" + displayBoard[3] + "\n" +
                " |  |  |     |  |  |\n" +
                " |  |  " + displayBoard[16] + "--" + displayBoard[17] + "--" + displayBoard[18] + "  |  |\n" +
                " |  |     |     |  |\n" +
                " |  " + displayBoard[8] + "-----" + displayBoard[9] + "-----" + displayBoard[10] + "  |\n" +
                " |        |        |\n" +
                " " + displayBoard[0] + "--------" + displayBoard[1] + "--------" + displayBoard[2];
    }


    public boolean isNonWinTerminal() {
        return pawnsToPlace == 0 && generateChildren().isEmpty();
    }

     public String stanPlanszy(){
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        for(int i=0;i<24;i++){
            if(board[i] == 'W'){
                sb1.append(i).append(" ");
            }else if(board[i] == 'B'){
                sb2.append(i).append(" ");
            }
        }
        return sb1.toString() + "| " + sb2.toString() + "| ";
    }
}

class heurystyka extends StateFunction {
    @Override
    public double calculate(State s) {
        Mill mill = (Mill) s;
        int whitePawns = mill.whitePawnsCnt;
        int blackPawns = mill.blackPawnsCnt;
        if (blackPawns < 3) {
            return Double.POSITIVE_INFINITY;
        } else if (whitePawns < 3) {
            return Double.NEGATIVE_INFINITY;
        }
        return whitePawns - blackPawns;
    }
}

public class Main {
    static int D = 1;
    public static void expand(GameState s,int d, int[] cnt){
        if(d==cnt.length){
            return;
        }
        cnt[d]++;
        if(d == D ){
//            System.out.println(((Mill)s).stanPlanszy() + s.generateChildren().size());
        }
        for(GameState t:s.generateChildren()){
            expand(t,d+1,cnt);
        }
    }
    public static void expand(GameState s,int d){
        int[] cnt = new int[d];
        expand(s,0,cnt);
        for (int i=0; i<d; i++){
            System.out.println("Depth: " + i + ": " + cnt[i] + " ");
        }
    }
    public static void main(String[] args) {
        GameState mlynek = new Mill();
        GameSearchConfigurator configurator = new GameSearchConfigurator();
        configurator.setDepthLimit (1.5);
        GameSearchAlgorithm alg = new MinMax(new Mill(), configurator);
        Scanner scanner = new Scanner(System.in);


        ((Mill)mlynek).board[0] = 'W';
        ((Mill)mlynek).board[1] = 'B';
        ((Mill)mlynek).board[3] = 'W';
        ((Mill)mlynek).board[5] = 'B';
        ((Mill)mlynek).board[8] = 'B';
        ((Mill)mlynek).board[12] = 'W';
        ((Mill)mlynek).board[13] = 'W';
        ((Mill)mlynek).board[14] = 'W';
        ((Mill)mlynek).board[19] = 'W';
        ((Mill)mlynek).board[21] = 'B';
        ((Mill)mlynek).board[23] = 'W';
        ((Mill)mlynek).pawnsToPlace = 0;
        ((Mill) mlynek).whitePawnsCnt = 7;
        ((Mill) mlynek).blackPawnsCnt = 4;
        System.out.println(mlynek);
        expand(mlynek,6);


//        while (!mlynek.isWinTerminal() && !mlynek.isNonWinTerminal()) {
//            System.out.println("Aktualny stan planszy: \n" + mlynek.toString());
//            // Tura gracza
//            System.out.println("Twoja tura: ");
//            String ruch = scanner.nextLine();
//            List<GameState> children = mlynek.generateChildren();
//            boolean validMove = false;
//            for (GameState child : children) {
//                if (ruch.equals(child.getMoveName())) {
//                    System.out.println(child.getMoveName());
//                    mlynek = child;
//                    validMove = true;
//                    break;
//                }
//            }
//
//            // Sprawdzenie poprawności ruchu
//            if (!validMove) {
//                System.out.println("Błędny ruch, spróbuj ponownie.");
//                continue;
//            }
//
//            if (((Mill) mlynek).isMillFormed(Integer.parseInt(ruch))) {
//                children = new ArrayList<>(((Mill) mlynek).solveMill());
//                mlynek = children.get(0);
//            }else{
//                mlynek.setMaximizingTurnNow(!mlynek.isMaximizingTurnNow());
//            }
//
//            // Sprawdź, czy gra się zakończyła po ruchu gracza
//            if (mlynek.isWinTerminal() || mlynek.isNonWinTerminal()) {
//                break;
//            }
//            System.out.println("Aktualny stan planszy: \n" + mlynek.toString());
//
//            // Tura przeciwnika
//            System.out.println("Tura przeciwnika...");
//            children = mlynek.generateChildren();
//            alg.setInitial(mlynek);
//            alg.execute();
//            String bestMove = alg.getFirstBestMove();
//            System.out.println("Przeciwnik wybrał ruch: " + bestMove);
//            for (GameState child : children) {
//                if (bestMove.equals(child.getMoveName())) {
//                    mlynek = child;
//                    break;
//                }
//            }
//            if (((Mill) mlynek).isMillFormed(Integer.parseInt(bestMove))) {
//                children = new ArrayList<>(((Mill) mlynek).solveMill());
//                mlynek = children.get(0);
//            }else{
//                mlynek.setMaximizingTurnNow(!mlynek.isMaximizingTurnNow());
//            }
//        }
//
//        if (mlynek.isWinTerminal()) {
//            System.out.println("Koniec gry! Wygrałeś!");
//        } else if (mlynek.isNonWinTerminal()) {
//            System.out.println("Koniec gry! Przegrałeś.");
//        } else {
//            System.out.println("Koniec gry! Remis.");
//        }
//
//        scanner.close();
    }
}