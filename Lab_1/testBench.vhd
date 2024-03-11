LIBRARY IEEE;
USE IEEE.std_logic_1164.all;
USE IEEE.numeric_std.all;
USE IEEE.math_real.all;

-- this is a testbench for the 32x32-bit register file for a 32-bit MIPS microprocessor defined in regFile.vhd

ENTITY testBench IS
	-- numerical constants are defined using generics (to avoid magic nums)
	GENERIC(	
		addressBits_GEN  : INTEGER := 5;  -- 0 -> 31
		dataLength_GEN   : INTEGER := 32; -- 32-bit data
		numOfReg_GEN     : INTEGER := 32  -- number of registers = 2 ^ addressBits
	);	
	-- no ports to define here
END testBench;


ARCHITECTURE testBench_ARCH OF testBench IS
	-- define regFile as a component 
	COMPONENT regFile
		PORT(
		-- input signals
		RsSel_IN  : IN  STD_LOGIC_VECTOR((addressBits_GEN - 1) DOWNTO 0); -- read reg 1
		RtSel_IN  : IN  STD_LOGIC_VECTOR((addressBits_GEN - 1) DOWNTO 0); -- read reg 2
		RdSel_IN  : IN  STD_LOGIC_VECTOR((addressBits_GEN - 1) DOWNTO 0); -- write reg
		DataW_IN  : IN  STD_LOGIC_VECTOR((dataLength_GEN - 1) DOWNTO 0); -- write this data in Rd

		-- clock + control signals
		WrtEN     : IN  STD_LOGIC; -- write enable
		CLK       : IN  STD_LOGIC; -- clock (write on falling and read on rising)		

		-- output signals
		DataRs_OUT : OUT STD_LOGIC_VECTOR((dataLength_GEN - 1) DOWNTO 0); -- data written in Rs
		DataRt_OUT : OUT STD_LOGIC_VECTOR((dataLength_GEN - 1) DOWNTO 0)  -- data written in Rd
	);
	END COMPONENT;
	
	-- declare signals
	SIGNAL RsSel_IN_SIG : STD_LOGIC_VECTOR((addressBits_GEN - 1) DOWNTO 0);
	SIGNAL RtSel_IN_SIG : STD_LOGIC_VECTOR((addressBits_GEN - 1) DOWNTO 0);
	SIGNAL RdSel_IN_SIG : STD_LOGIC_VECTOR((addressBits_GEN - 1) DOWNTO 0);
	SIGNAL DataW_IN_SIG : STD_LOGIC_VECTOR((dataLength_GEN - 1) DOWNTO 0);
	
	SIGNAL WrtEN : STD_LOGIC := '0'; -- WrtEN has to have an initial value
	SIGNAL CLK   : STD_LOGIC := '0'; -- clk has to have an initial value
	
	SIGNAL DataRs_OUT_SIG : STD_LOGIC_VECTOR((dataLength_GEN - 1) DOWNTO 0);
	SIGNAL DataRt_OUT_SIG : STD_LOGIC_VECTOR((dataLength_GEN - 1) DOWNTO 0);
	
	-- declare constant clock period
	CONSTANT clkPeriod_CON : TIME := 100 ps;
	
BEGIN
	-- mapping signals to component ports
	Test1: regFile PORT MAP (
		RsSel_IN 	=> RsSel_IN_SIG,
		RtSel_IN 	=> RtSel_IN_SIG,
		RdSel_IN 	=> RdSel_IN_SIG,
		DataW_IN 	=> DataW_IN_SIG,
		WrtEN    	=> WrtEN,
		CLK		=> CLK,
		DataRs_OUT	=> DataRs_OUT_SIG,
		DataRt_OUT	=> DataRt_OUT_SIG
	);
	-- testBench process for CLK		
	PROCESS(CLK) IS
	BEGIN	
		CLK   <= NOT CLK AFTER (clkPeriod_CON / 2);
	END PROCESS;
	-- testBench process for WrtEN
	PROCESS(WrtEN) IS
		CONSTANT clkCyclesNum_GEN : SIGNED(3 DOWNTO 0) := "1010"; -- number of rising/falling edges during this test
	BEGIN
		--WrtEN <= NOT WrtEN AFTER ((clkPeriod_CON / 2) * clkCyclesNum_GEN);
		WrtEN <= NOT WrtEN AFTER 500 ps;
	END PROCESS;
	-- set values for the rest of inputs
	RsSel_IN_SIG <= "00111";
	RtSel_IN_SIG <= "01111";
	RdSel_IN_SIG <= "10000";
	DataW_IN_SIG <= X"AAAAAAAA";
END testBench_ARCH;



/* SEQUENCER_PROC : process      

    procedure test (x : integer; y : integer) is
    begin
      a <= x;
      b <= y;
      wait for 10 ns;
      
      report "a = " & integer'image(a)
        & ", b = " & integer'image(b)
        & ", q = " & boolean'image(q);
    end procedure;

  begin

    test(10, 5);
    test(5, 5);
    test(5, 10);

    wait;
  end process; */