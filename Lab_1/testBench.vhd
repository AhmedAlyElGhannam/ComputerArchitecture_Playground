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
	-- signal for test cases
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
	Test1: 
	regFile PORT MAP (
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
	clkCycleProcess : 
	PROCESS(CLK)
	BEGIN	
		CLK <= NOT CLK AFTER (clkPeriod_CON / 2);
	END PROCESS clkCycleProcess;
	-- testBench process for WrtEN
	writeEnableProcess : 
	PROCESS(WrtEN)
	BEGIN
		-- flip WrtEn every 200 ps  
		WrtEN <= NOT WrtEN AFTER 2 * clkPeriod_CON;
	END PROCESS writeEnableProcess;
	-- testBench process for test cases
	testCaseProcess:
	PROCESS 
		-- define procedure to set input signals values and wait
		PROCEDURE regFileInputsTest(
			CONSTANT RsSel_IN_PROC   		: STD_LOGIC_VECTOR((addressBits_CON - 1) DOWNTO 0);
			CONSTANT RtSel_IN_PROC   		: STD_LOGIC_VECTOR((addressBits_CON - 1) DOWNTO 0);
			CONSTANT RdSel_IN_PROC   		: STD_LOGIC_VECTOR((addressBits_CON - 1) DOWNTO 0);
			CONSTANT DataW_IN_PROC   		: STD_LOGIC_VECTOR((dataLength_CON - 1) DOWNTO 0);
			CONSTANT testCaseDuration_PROC 	: TIME
		) IS
		BEGIN
			-- set inputs by passed values
			RsSel_IN_SIG <= RsSel_IN_PROC;
			RtSel_IN_SIG <= RtSel_IN_PROC;
			RdSel_IN_SIG <= RdSel_IN_PROC;
			DataW_IN_SIG <= DataW_IN_PROC;
			-- wait to view results of this test case
			WAIT FOR testCaseDuration_PROC;
		END PROCEDURE regFileInputsTest;
	BEGIN
		-- Initial wait duration before doing anything
		WAIT FOR clkPeriod_CON / 4;
		-- Test case 1:
		---- -> read rs && rt && try to write a diff value in rd (will not write successfully until WrtEN is 1)
		---- -> Rs=7,Rt=8,Rd=9,DataWrite=X"AAAAAAAA",Duration=400ps (4 clk cycles)
		regFileInputsTest("00111","01000","01001",X"AAAAAAAA",400 ps);
		-- Test case 2:
		---- -> read $0 as rs and read $9 as rt then try to write some value in rs ($0 value should remain 0)
		---- -> Rs=0,Rt=9,Rd=0,DataWrite=X"ABCDDCBA",Duration=400ps (4 clk cycles)
		regFileInputsTest("00000","01001","00000",X"ABCDDCBA",400 ps);
		-- Test case 3:
		---- -> read what is in rs and write in rd (first clock cycle should show read and second should show write)
		---- -> Rs=12,Rt=0,Rd=12,DataWrite=X"EEEEEEEE",Duration=400ps (4 clk cycles)
		regFileInputsTest("01100","00000","01100",X"EEEEEEEE",400 ps);
	END PROCESS testCaseProcess;
END testBench_ARCH;