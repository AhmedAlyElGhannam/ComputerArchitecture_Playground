LIBRARY IEEE;
USE IEEE.std_logic_1164.all;
USE IEEE.numeric_std.all;
USE IEEE.math_real.all;

-- this is a testbench for the 32x32-bit register file for a 32-bit MIPS microprocessor defined in regFile.vhd

ENTITY testBench IS
	-- nothing to see here
END testBench;


ARCHITECTURE testBench_ARCH OF testBench IS
	-- declare constant clock period
	CONSTANT clkPeriod_CON 	  : TIME    := 100 ps;
	-- define number of address bits 
	CONSTANT addressBits_CON  : INTEGER := 5;  
	-- define number of 32-bit data
	CONSTANT dataLength_CON   : INTEGER := 32; 
	-- number of registers = 2 ^ addressBits
	CONSTANT numOfReg_CON     : INTEGER := 32; 
	-- variable for test cases
	--VARIABLE testCase_VAR	  : INTEGER := 1;
	-- define regFile as a component 
	COMPONENT regFile
		PORT(
		-- input ports
		RsSel_IN  : IN  STD_LOGIC_VECTOR((addressBits_CON - 1) DOWNTO 0); -- read reg 1
		RtSel_IN  : IN  STD_LOGIC_VECTOR((addressBits_CON - 1) DOWNTO 0); -- read reg 2
		RdSel_IN  : IN  STD_LOGIC_VECTOR((addressBits_CON - 1) DOWNTO 0); -- write reg
		DataW_IN  : IN  STD_LOGIC_VECTOR((dataLength_CON - 1) DOWNTO 0); -- write this data in Rd

		-- clock && necessary control ports(s)
		WrtEN     : IN  STD_LOGIC; -- write enable
		CLK       : IN  STD_LOGIC; -- clock (write on falling and read on rising)		

		-- output ports
		DataRs_OUT : OUT STD_LOGIC_VECTOR((dataLength_CON - 1) DOWNTO 0); -- data written in Rs
		DataRt_OUT : OUT STD_LOGIC_VECTOR((dataLength_CON - 1) DOWNTO 0)  -- data written in Rd
	);
	END COMPONENT;
	-- input signals
	SIGNAL RsSel_IN_SIG : STD_LOGIC_VECTOR((addressBits_CON - 1) DOWNTO 0);
	SIGNAL RtSel_IN_SIG : STD_LOGIC_VECTOR((addressBits_CON - 1) DOWNTO 0);
	SIGNAL RdSel_IN_SIG : STD_LOGIC_VECTOR((addressBits_CON - 1) DOWNTO 0);
	SIGNAL DataW_IN_SIG : STD_LOGIC_VECTOR((dataLength_CON - 1) DOWNTO 0);
	-- control && clk signal
	SIGNAL WrtEN : STD_LOGIC := '0'; -- WrtEN has to have an initial value
	SIGNAL CLK   : STD_LOGIC := '0'; -- clk has to have an initial value
	-- output signals 
	SIGNAL DataRs_OUT_SIG : STD_LOGIC_VECTOR((dataLength_CON - 1) DOWNTO 0);
	SIGNAL DataRt_OUT_SIG : STD_LOGIC_VECTOR((dataLength_CON - 1) DOWNTO 0);
BEGIN
	-- mapping signals to component ports
	Test1: regFile PORT MAP (
		RsSel_IN 	=> RsSel_IN_SIG,
		RtSel_IN 	=> RtSel_IN_SIG,
		RdSel_IN 	=> RdSel_IN_SIG,
		DataW_IN 	=> DataW_IN_SIG,
		WrtEN    	=> WrtEN,
		CLK			=> CLK,
		DataRs_OUT	=> DataRs_OUT_SIG,
		DataRt_OUT	=> DataRt_OUT_SIG
	);
	-- testBench process for CLK		
	clkCycleProcess : PROCESS(CLK)
		VARIABLE clkCycleCounter_VAR : INTEGER := 1;
	BEGIN	
		CLK <= NOT CLK AFTER (clkPeriod_CON / 2);
		CASE clkCycleCounter_VAR IS
			WHEN 1 =>
				-- set values for test case 1
				RsSel_IN_SIG <= "00111";
				RtSel_IN_SIG <= "01111";
				RdSel_IN_SIG <= "00111";
				DataW_IN_SIG <= X"AAAAAAAA";
				clkCycleCounter_VAR := clkCycleCounter_VAR + 1;
				--WAIT FOR 200 ps;
			WHEN 2 =>
				-- set values for test case 2
			WHEN 3 =>
				-- set values for test case 3
			WHEN 4 =>
				-- set values for test case 4
			WHEN 5 =>
				-- set values for test case 5
			WHEN OTHERS =>
				-- do nothing  
		END CASE;
	END PROCESS clkCycleProcess;
	-- testBench process for WrtEN
	writeEnableProcess : PROCESS(WrtEN)
	BEGIN
		-- flip WrtEn every 100 ps  
		WrtEN <= NOT WrtEN AFTER 300 ps;
	END PROCESS writeEnableProcess;
END testBench_ARCH;

